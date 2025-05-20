from typing import List, Dict, Any, Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import Graph, StateGraph
from langchain_openai import ChatOpenAI
from dotenv import dotenv_values
from textwrap import dedent

from config.prompts import (
    PLANNING_SYSTEM_PROMPT,
    EXECUTION_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT
)

from tools.new_stock_tools import StockAnalysis
from tools.agert_tools import LangGraphToolConverter

from langgraph.config import get_stream_writer

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
deepseek = True

if deepseek:
    my_api_key = config.get("DEEPSEEK_MODEL_API_KEY")
    model = ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key=my_api_key,
            streaming=True
        ).bind_tools(tools)


    summary_model = ChatOpenAI(
        model="deepseek-reasoner",
        base_url="https://api.deepseek.com/v1",
        api_key=my_api_key,
        streaming=True
    )
else:
    my_api_key = config.get("MODEL_API_KEY")
    model = ChatOpenAI(
            model="qwen-max-latest",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",
            api_key=my_api_key,
            streaming=True
        ).bind_tools(tools)


    summary_model = ChatOpenAI(
        model="qwq-plus-latest",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",
        api_key=my_api_key,
        streaming=True
    )

reasoner_model = ChatOpenAI(
    model="deepseek-reasoner",
    base_url="https://api.deepseek.com/v1",
    api_key=my_api_key,
    streaming=True,
    use_responses_api=True, 
    )

class AgentState(TypedDict):
    messages: List[BaseMessage]
    plan: List[str]
    current_step: int
    results: List[str]
    final_answer: str

def create_plan(state: AgentState) -> AgentState:
    """根据问题生成执行计划"""
    messages = state["messages"]
    
    # 提取最后一条用户消息中的问题和思维导图
    last_message = messages[-1].content
    
    # 构建计划生成提示
    planning_prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNING_SYSTEM_PROMPT),
        ("user", "{input}")
    ])
    
    # 生成计划
    response = summary_model.invoke(
        planning_prompt.format_messages(input=last_message)
    )
    #使用正则匹配"<"到"]"之前的内容，解析计划
    plan_steps = response.content.split



    # 按照<>[]解析生成的计划
    plan_steps = response.content.split("\n")
    for i in range(len(plan_steps)):
        # 去除每个步骤的前后空格
        plan_steps[i] = plan_steps[i].strip()
        #过滤掉不含步骤的内容
        if (not plan_steps[i].startswith("<")) and (not plan_steps[i].endswith("]")):
            #删除步骤
            plan_steps[i] = ""
            
    
    # 过滤掉空步骤
    plan_steps = [step for step in plan_steps if step]
    

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
    writer = get_stream_writer()
    """执行当前计划步骤"""
    if state["current_step"] >= len(state["plan"]):
        return state
    
    current_step = state["plan"][state["current_step"]]
    writer(f"> 当前步骤: {current_step}\n\n")
    messages = state["messages"]

    #将之前步骤执行结果添加到文本中
    plan_result = ""
    for i in range(state["current_step"]):
        plan_result += f"{state['plan'][i]} "
        plan_result += f"执行结果：{state['results'][i]} \n"

    plan_result = dedent(plan_result)

    # 构建执行步骤的提示
    execution_prompt = ChatPromptTemplate.from_messages([
        ("system", "{plan_result}"),
        ("system", EXECUTION_SYSTEM_PROMPT),
        ("user", "{question}当前步骤: {current_step}")
    ])

    input = execution_prompt.format_messages(
            question=messages[0].content,
            current_step=current_step,
            plan_result=plan_result
    )

    #判断步骤类型
    if current_step.startswith("<数据>"):
        # current_step = current_step.replace("<数据>", "").strip()
        # 执行步骤
        response = model.invoke(input)
        res = []
        #判断是否调用工具
        if response.tool_calls:
            for toolcall in response.tool_calls:
                tool = find_tool_by_name(toolcall.get("name"))
                if tool:
                    # 调用工具
                    tool_response = tool.invoke(toolcall.get("args"))
                    # 将工具的响应添加到结果中
                    res.append(tool_response)
                else:
                    print(f"未找到工具: {toolcall.get('name')}")

    elif current_step.startswith("<分析>"):
        # current_step = current_step.replace("<分析>", "").strip()
        # 执行步骤
        response = model.invoke(input)
        res = response.content
    else:
        res = ""

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
    writer = get_stream_writer()

    #将之前步骤执行结果添加到文本中
    plan_result = ""
    for i in range(state["current_step"]):
        plan_result += f"{state['plan'][i]} "
        plan_result += f"执行结果：{state['results'][i]} \n"

    plan_result = dedent(plan_result)
    
    # 构建总结提示
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", SUMMARY_SYSTEM_PROMPT),
        ("user", "原始问题: {question}\n\n计划及执行结果:\n{results}")
    ])

    response = summary_model.stream(
        summary_prompt.format_messages(
            question=messages[0].content,
            results=plan_result
        )
    )
    ai_msg = ""
    for chunk in response:
        if chunk.content != '':
            writer(chunk.content)
            ai_msg += chunk.content.strip()
    
    return {
        **state,
        "final_answer": ai_msg,
        "results": state["results"] + [ai_msg],
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

def run_agent(question: str) :
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
    final_state = agent.stream(initial_state,config={"callbacks": [langfuse_handler]},stream_mode = "custom")
    for chunk in final_state:
        yield chunk
