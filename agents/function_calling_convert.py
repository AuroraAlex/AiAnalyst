from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from dotenv import dotenv_values
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from tools.new_stock_tools import StockAnalysis

stock = StockAnalysis()
# stock_date_df = stock.get_stock_daily_data("80020", "XHKG", 120)
stock_date_df = stock.get_stock_daily_data("TSLA", "XNAS", 120)

@tool
def get_stock_ema(period: int) -> str:
    """
    获取股票的EMA指标数据
    Args:
        period: EMA的计算周期
    Returns:
        EMA指标数据(JSON格式)
    """

    RS_DF = stock.calculate_ema(stock_date_df,period)
    ret = stock.pd_json(RS_DF)
    return ret

@tool
def get_120_days_data() -> str:
    """
    获取股票的120天日K数据
    Returns:
        120天日K数据(JSON格式) 
    """

    ret = stock.pd_json(stock_date_df)
    return ret

@tool
def get_stock_macd(fast_period: int = 12,slow_period: int = 26,signal_period: int = 9) -> str:
    """
    获取股票的MACD指标数据
    Args:
        fast_period: MACD的快线周期
        slow_period: MACD的慢线周期
        signal_period: MACD的信号线周期
    Returns:
        MACD指标数据(JSON格式)
    """

    RS_DF = stock.calculate_macd(stock_date_df,fast_period,slow_period,signal_period)
    ret = stock.pd_json(RS_DF)
    return ret

@tool
def get_stock_rsi(period: int = 14) -> str:
    """
    获取股票的RSI指标数据
    Args:
        period: RSI的计算周期
    Returns:
        RSI指标数据(JSON格式)
    """

    RS_DF = stock.calculate_rsi(stock_date_df,period)
    ret = stock.pd_json(RS_DF)
    return ret

@tool
def get_stock_sma(period: int = 20) -> str:
    """
    获取股票的SMA指标数据
    Args:
        period: SMA的计算周期
    Returns:
        SMA指标数据(JSON格式)
    """

    RS_DF = stock.calculate_sma(stock_date_df,period)
    ret = stock.pd_json(RS_DF)
    return ret

@tool
def get_stock_bollinger_bands(period: int = 20, std_dev: int = 2) -> str:
    """
    获取股票的布林带指标数据
    Args:
        period: 布林带的计算周期
        std_dev: 布林带的标准差
    Returns:
        布林带指标数据(JSON格式)
    """

    RS_DF = stock.calculate_bollinger_bands(stock_date_df,period,std_dev)
    ret = stock.pd_json(RS_DF)
    return ret

@tool
def get_stock_kdj(k_period: int = 14, d_period: int = 3) -> str:
    """
    获取股票的KDJ指标数据
    Args:
        k_period: KDJ的K线计算周期
        d_period: KDJ的D线计算周期
    Returns:
        KDJ指标数据(JSON格式)
    """

    RS_DF = stock.calculate_kdj(stock_date_df,k_period,d_period)
    ret = stock.pd_json(RS_DF)
    return ret

@tool
def get_stock_atr(period: int = 14) -> str:
    """
    获取股票的ATR指标数据
    Args:
        period: ATR的计算周期
    Returns:
        ATR指标数据(JSON格式)
    """

    RS_DF = stock.calculate_atr(stock_date_df,period)
    ret = stock.pd_json(RS_DF)
    return ret

@tool
def get_stock_obv() -> str:
    """
    获取股票的OBV指标数据
    Returns:
        OBV指标数据(JSON格式)
    """

    RS_DF = stock.calculate_obv(stock_date_df)
    ret = stock.pd_json(RS_DF)
    return ret

class FunctionCallingAgent:
    """
    技术指标分析Agent，可以通过接口获取指定时间段的日K数据并计算技术指标
    支持Function Calling让大模型自行决定调用哪些工具
    """
    def __init__(self):
         # 加载 .env 文件
        load_dotenv()
        self.model_api_key = os.getenv("MODEL_API_KEY")
        self.data_base_url = os.getenv("MODEL_BASE_URL")
        self.model_name = os.getenv("INDICATORS_MODEL_NAME")
        # 修改为非流模式，解决兼容性问题
        self.llm = ChatOpenAI(
            model_name = self.model_name,
            openai_api_key = self.model_api_key,
            openai_api_base = self.data_base_url,
            stream = True  # 修改为 False 解决流式输出的兼容性问题
        )


    def creat_agent(self):
        #绑定工具
        tools = [get_120_days_data,get_stock_ema,get_stock_macd,get_stock_rsi,get_stock_sma,get_stock_bollinger_bands,get_stock_kdj,get_stock_atr,get_stock_obv]
        self.model_with_tools = self.llm.bind_tools(tools)
        # 创建工作流
        workflow = StateGraph(MessagesState)
        tool_node = ToolNode(tools)

        def call_model(state: MessagesState):
            messages = state["messages"]
            response = self.model_with_tools.invoke(messages)
            return {"messages": [response]}
        
        def should_continue(state: MessagesState):
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                return "tools"
            return END
        
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, ["tools", END])
        workflow.add_edge("tools", "agent")
        app = workflow.compile()
        return app


