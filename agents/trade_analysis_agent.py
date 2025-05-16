from typing import List, Dict, Any, Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import Graph, StateGraph
from langchain_openai import ChatOpenAI
from dotenv import dotenv_values
from textwrap import dedent

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
    analysis_tool.get_real_time_data,
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
        ("system", """
# 技能\n\n
你是一个金融分析师，你需要判断对股票进行技术面分析需要哪些步骤，你需要生成一系列的执行计划，用于获取数据以及技术指标，最终结合数据指标生成一个低频量化的分析报告帮助决策。
# 要求\n\n
我会提供一系列的工具，你可以使用这些工具来获取数据和分析数据。\n\n
<>内表示需要执行步骤的类型，目前需要规划的步骤有两种类型：
    1. <数据>：获取股票的历史数据，或者获取其他相关数据,可以使用一个或多个工具,你需要指定获取多少天的日线数据，需要注意计算技术指标工具的周期必须远小于获取日线数据的周期，因为指定周期内存在节假日不开盘。同时获取到实时行情（盘口数据）以便更准确的分析买卖时机。
    2. <总结>：对分析结果进行深度总结，生成最终的回答
\n\n[]内表示需要执行的计划内容。
    每个步骤都应该清晰具体，步骤间使用"\n"进行分隔，以便解析。一个计划可以包含多个内容，尽可能的执行少的步骤完成任务。
    "<>"、"[]"内部禁止使用换行或"\n"。
    目前无法获取除问题中提供的股票代码和交易所代码以外的其他数据。
    # 示例\n\n
    以下是一个示例：
    <数据>[获取特斯拉的历史数据]
    <数据>[计算特斯拉的移动平均线、KDJ、RSI]
    <总结>[生成分析报告]
    
         """
         ),
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
        ("system", 
         """
        根据当前步骤
        执行任务，如果需要可以使用可用的工具。通过工具获取的数据的索引越大时间越接近现在。
        <>内表示需要执行步骤的类型，目前有三种类型：
            1. <数据>：获取股票的历史数据，或者获取其他相关数据,可以使用一个或多个工具
            2. <总结>：对分析结果进行深度总结，不能调用工具，生成最终的回答
        []内表示需要执行的计划内容。
         """),
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
        ("system", """
         # 技能\n\n
         你是一个金融分析师，你需要结合技术指标对股票进行深度分析，生成一个便于低频量化交易（交易频率可以根据你的分析决定）的分析报告。
         工具获取的数据和分析结果已经完成，数据的索引越大时间越接近现在，接下来你需要对这些数据进行深度总结，生成最终的回答。
        # 输出内容\n\n
        以下为必须包含的内容：
            1.该股票是否值得进行投资，是否值得进行低频量化交易。
            2.结合技术指标的分析结果，判断股票的买入和卖出时机（不需要最理想的买入卖出价格，只追求最可能的买入卖出价格），以及止损和止盈策略。
            3.给出买卖的周期建议。
        # 示例\n\n
        以下为示例内容：
            # 股票代码：AAPL
            # 交易所：NASDAQ
            # 交易周期：3 天
            # 当前价格：**150.00** 建议投资指数（0-5）：** 4 ** 风险指数：** 2 **
            # 技术指标分析：\n\n
            ···（此处自由发挥）
            # 买入卖出时机：\n\n
            - 不建议进行交易（若此时机不合适）
            - 买入：建议在价格回调到**145.00**时进行买入操作。（不建议进行交易时忽略）
            - 卖出：建议在价格上涨到**155.00**时进行卖出操作。（不建议进行交易时忽略）
         
            # 止盈止损策略（不建议进行交易时忽略）：
            - 止盈：建议在价格达到**160.00**时进行止盈操作。
            - 止损：建议在价格跌破**140.00**时进行止损操作。
            # 预期\n\n
            - 预期盈利空间：**6.00%**
            - 预期最大回撤空间：**5.00%**
            - 预期收益风险比：**1.20**
            # 其他建议：\n\n
            ...(此处自由发挥)
            
         """),
        ("user", "原始问题: {question}\n\n计划及执行结果:\n{results}"),
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
