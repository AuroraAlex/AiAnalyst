from typing import List, Dict, Any, Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import Graph, StateGraph
from langchain_openai import ChatOpenAI
from dotenv import dotenv_values
from textwrap import dedent

import os
import datetime
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

def get_analysis_result(stock_name: str, stock_code: str, exchange_code: str) -> str:
    """
    获取股票的分析结果
    """
    analysis_result = {}
    #根据股票代码和交易所代码获取分析结果所在的文件夹
    analysis_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", f"{stock_name}_{stock_code}_{exchange_code}")
    #检查文件夹是否存在
    if not os.path.exists(analysis_folder):
        raise FileNotFoundError(f"Analysis folder {analysis_folder} does not exist.")
    #获取文件夹下所有文件，文件名格式为 %Y%m%d%H.md

    files = [f for f in os.listdir(analysis_folder) if f.endswith(".md")]
    #%Y%m%d%H为文件名格式，将字符串转换为 datetime 对象，并按照日期排序
    files_dates = [datetime.datetime.strptime(f, "%Y%m%d%H.md") for f in files]
    # 按照日期排序
    files_dates.sort(reverse=True)
    # 获取最新的文件
    latest_file = files_dates[0]
    # 获取最新文件的路径
    latest_file_path = os.path.join(analysis_folder, latest_file.strftime("%Y%m%d%H.md"))
    # 读取最新文件的内容
    with open(latest_file_path, "r") as f:
        analysis_result = f.read()
    return analysis_result

def check_strategy(stock_name: str, stock_code: str, exchange_code: str):
    """获取指定股票的分析结果"""
    analysis_result = get_analysis_result(stock_name, stock_code, exchange_code)
    pass

