from typing import List, Dict, Any, Optional, Callable, Union
import base64
from pathlib import Path
import os
import json
import inspect
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from langchain_core.messages import ToolMessage
import re

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 然后保持原导入不变
from tools.stock_tools import StockQuery



class IndicatorsAnalysisAgent:
    """
    技术指标分析Agent，可以通过接口获取指定时间段的日K数据并计算技术指标
    支持Function Calling让大模型自行决定调用哪些工具
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
         # 初始化Stock查询工具
        self.stock_query = StockQuery(config=self.config)
        self._initialize_model()
       
        
    
    def _initialize_model(self):
        """Initialize the LangChain chat model with configuration."""
        api_keys = self.config.get("api_keys", {})
        model_config = self.config.get("models", {}).get("indicators_analysis", {})
        
        self.llm = ChatOpenAI(
            model_name=model_config.get("model_name"),
            openai_api_key=api_keys.get("bailian_api_key"),
            openai_api_base=model_config.get("base_url")
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="indicator_chat_history",
            return_messages=True
        )

        # 创建工具列表
        self.tools = self._create_tools()
        
    def _create_tools(self) -> List[Tool]:
        """
        创建可用于Function Calling的工具列表
        
        Returns:
            工具列表
        """
        tools = []
        
        # 获取指定股票在指定时间段内的技术指标
        class GetIndicatorsInput(BaseModel):
            stock_code: str = Field(..., description="股票代码，例如'AAPL'")
            exchange_code: str = Field(..., description="交易所代码，例如'XNAS'")
            days: int = Field(90, description="获取的天数，默认90天")
            
        def get_indicators_tool(stock_code: str, exchange_code: str, days: int = 90) -> Dict:
            """获取指定股票在指定时间段内的技术指标"""
            return self.stock_query.get_stock_daily_data(stock_code, exchange_code, days)
            
        tools.append(
            Tool.from_function(
                func=get_indicators_tool,
                name="get_stock_indicators",
                description="获取指定股票在指定时间段内的技术指标",
                args_schema=GetIndicatorsInput
            )
        )
        return tools

# 示例用法
if __name__ == "__main__":
    # 初始化Agent
    agent = IndicatorsAnalysisAgent()

    tool_name = [tool["function"]["name"] for tool in agent.tools]
    print(f"创建了{len(agent.tools)}个工具，为：{tool_name}\n")
    
    # # 使用Function Calling处理用户查询
    # result = agent.process_query("帮我分析一下特斯拉(TSLA)最近90天的股票趋势，我想了解MACD和RSI指标的信号")
    # print(result)
    
    # # 另一个使用Function Calling的例子
    # result = agent.process_query("请获取苹果公司最近120天的股票技术指标，并保存到apple_analysis.json文件")
    # print(result)
