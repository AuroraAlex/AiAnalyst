from typing import List, Dict, Any, Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import Graph, StateGraph
from langchain_openai import ChatOpenAI
from dotenv import dotenv_values
from textwrap import dedent

import json

from tools.new_stock_tools import StockAnalysis
from tools.agert_tools import LangGraphToolConverter

from langgraph.config import get_stream_writer

from langfuse.callback import CallbackHandler


def get_stock_pool():
    """
    读取股票池、分析每只股票的技术面
    """
    stock_list = []

    with open("../config/stock_pool.json", "r") as f:
        data = json.load(f)
        

    return stock_list

def check_strategy():
    pass

if __name__ == "__main__":
    pass

