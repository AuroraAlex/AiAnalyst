from typing import List, Dict, Any, Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import Graph, StateGraph
from langchain_openai import ChatOpenAI
from dotenv import dotenv_values

from tools.new_stock_tools import StockAnalysis
from tools.agert_tools import LangGraphToolConverter

from langfuse.callback import CallbackHandler
# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()

converteor = LangGraphToolConverter()
analysis_tool = StockAnalysis()
# 创建工具

tools = LangGraphToolConverter().functions_to_tools([
    analysis_tool.get_stock_daily_data,
    analysis_tool.calculate_ema,
    analysis_tool.calculate_sma,
    analysis_tool.calculate_macd,
    analysis_tool.calculate_rsi,
    analysis_tool.calculate_bollinger_bands,
    analysis_tool.calculate_atr,
    analysis_tool.calculate_obv,
    analysis_tool.calculate_kdj,
])


config = dotenv_values("./.env")
my_api_key = config.get("MODEL_API_KEY")

model = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=my_api_key,
        streaming=True
    ).bind_tools(tools)


class AgentState(TypedDict):
    messages: List[BaseMessage]
    plan: List[str]
    current_step: int
    results: List[str]
    final_answer: str

def create_plan(state: AgentState) -> AgentState:
    """根据问题和思维导图生成执行计划"""
    messages = state["messages"]
    
    # 提取最后一条用户消息中的问题和思维导图
    last_message = messages[-1].content
    
    # 构建计划生成提示
    planning_prompt = ChatPromptTemplate.from_messages([
        ("system", """
         你是一个金融分析师，你需要对指定的股票进行分析，首先你需要生成一个详细的执行计划，用于获取数据，最终结合数据来分析股票。
         我会提供一系列的工具，你可以使用这些工具来获取数据和分析数据。
         根据用户的问题，生成一个详细的执行计划。每个步骤都应该清晰具体，通过“<>”来包扩内容，步骤间使用换行进行分隔，以便我解析。
         同一个步骤可以执行多个工具，尽可能的执行少的步骤。
         以下是一个示例：
            < 步骤 1： 获取特斯拉的历史数据 >
            < 步骤 2： 计算特斯拉的移动平均线、KDJ、RSI >
            < 步骤 3： 综合技术指标和历史数据，分析特斯拉的股票走势 >
            < 步骤 4： 生成分析报告 >
         """
         ),
        ("user", "{input}")
    ])
    
    # 生成计划
    response = model.invoke(
        planning_prompt.format_messages(input=last_message)
    )
    
    # 解析生成的计划,每个步骤被<>包围
    # 这里假设返回的内容是以换行符分隔的步骤
    # 例如："< 步骤 1： >\n< 步骤 2： >\n< 步骤 3： >"
    plan_steps = response.content.split("\n")
    plan_steps = [step.strip("< >") for step in plan_steps if step.strip()]
    

    print("Generated plan steps:", plan_steps)
    
    return {
        **state,
        "plan": plan_steps,
        "current_step": 0,
        "results": []
    }

def find_tool_by_name(tools_name: str):
    """通过名称查找工具"""
    for tool in tools:
        if tool.name == tools_name:
            return tool
    return None
    
def execute_step(state: AgentState) -> AgentState:
    """执行当前计划步骤"""
    if state["current_step"] >= len(state["plan"]):
        return state
        
    current_step = state["plan"][state["current_step"]]
    messages = state["messages"]
    
    # 构建执行步骤的提示
    execution_prompt = ChatPromptTemplate.from_messages([
        ("system", "根据当前步骤执行任务，如果需要可以使用可用的工具。"),
        ("user", "{context}\n\n当前步骤: {step}")
    ])
    
    # 执行步骤
    response = model.invoke(
        execution_prompt.format_messages(
            context=messages[-1].content,
            step=current_step
        )
    )

    #判断是否调用工具
    if response.tool_calls:
        res = []
        for toolcall in response.tool_calls:
            tool = find_tool_by_name(toolcall.get("name"))
            if tool:
                # 调用工具
                tool_response = tool.invoke(toolcall.get("args"))
                # 将工具的响应添加到结果中
                res.append(tool_response)
                # 将工具的响应添加到消息中
                messages.append(AIMessage(content=tool_response))
            else:

                messages.append(AIMessage(content=f"未找到工具: {toolcall.get('name')}"))
    else:
        # 如果没有工具调用，分析结果
        tools_res = state["results"]
        res = model.invoke(
            execution_prompt.format_messages(
                context=messages[-1].content,
                step=current_step
            )
        )
        res = res.content
        messages.append(AIMessage(content=res))
    return {
        **state,
        "current_step": state["current_step"] + 1,
        "results": state["results"] + [str(res)],
    }

def should_continue(state: AgentState) -> str:
    """检查是否需要继续执行下一个步骤"""
    if state["current_step"] < len(state["plan"]):
        return "continue"
    return "finish"

def generate_final_answer(state: AgentState) -> AgentState:
    """生成最终答案"""
    messages = state["messages"]
    results = state["results"]
    
    # 构建总结提示
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "根据执行结果生成一个完整的回答。"),
        ("user", "原始问题: {question}\n\n执行结果:\n{results}")
    ])
    
    # 生成总结
    summary_model = ChatOpenAI(
        model="deepseek-reasoner",
        base_url="https://api.deepseek.com/v1",
        api_key=my_api_key,
        streaming=True
    )

    response = summary_model.invoke(
        summary_prompt.format_messages(
            question=messages[0].content,
            results="\n".join(results)
        )
    )
    
    return {
        **state,
        "final_answer": response.content
    }

def create_agent() -> Graph:
    """创建工作流图"""
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("execute_step", execute_step)
    workflow.add_node("generate_final_answer", generate_final_answer)
    
    # 设置入口节点
    workflow.set_entry_point("create_plan")
    
    # 添加边和条件
    workflow.add_edge("create_plan", "execute_step")
    workflow.add_conditional_edges(
        "execute_step",
        should_continue,
        {
            "continue": "execute_step",
            "finish": "generate_final_answer"
        }
    )
    
    # 设置出口节点
    workflow.set_finish_point("generate_final_answer")
    
    return workflow.compile()

def run_agent(question: str) -> str:
    """运行 Agent 处理问题"""
    # 组合问题和思维导图
    input_message = f"问题：{question}\n\n"
    
    # 初始化状态
    initial_state = {
        "messages": [HumanMessage(content=input_message)],
        "plan": [],
        "current_step": 0,
        "results": [],
        "final_answer": ""
    }
    
    # 创建并运行工作流
    agent = create_agent()
    #使用langfuse进行trace
    final_state = agent.invoke(initial_state,config={"callbacks": [langfuse_handler]})
    
    return final_state["final_answer"]

if "__main__" == __name__:
    question = "如何实现一个简单的待办事项应用？"
    mindmap = """
    graph TD
        A[待办事项应用] --> B[前端界面]
        A --> C[数据存储]
        B --> D[添加任务]
        B --> E[显示任务列表]
        B --> F[标记完成状态]
        C --> G[本地存储]
        C --> H[数据同步]
    """

    result = run_agent(question, mindmap)
    print(result)