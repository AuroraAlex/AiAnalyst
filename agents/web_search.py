from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from duckduckgo_search import DDGS
from dotenv import dotenv_values
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import os
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END


from langgraph.prebuilt import ToolNode

# 加载环境变量
config = dotenv_values("../.env")
my_api_key = config.get("OPENAI_API_KEY")

# 配置代理
os.environ['http_proxy'] = 'http://127.0.0.1:2561'
os.environ['https_proxy'] = 'http://127.0.0.1:2561'

# 创建搜索工具
@tool
def web_search(query: str) -> str:
    """搜索网络内容。当你需要查找最新信息或特定事实时使用此工具。

    Args:
        query: 搜索查询字符串

    Returns:
        搜索结果的摘要
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        if not results:
            return "没有找到相关结果。"
    
        summary = "\n\n".join(f"- {result['title']}\n{result['body']}" for result in results)
        return results


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# 测试 Agent
if __name__ == "__main__":
    tools = [web_search]
    tool_node = ToolNode(tools)
    # 创建 ChatOpenAI 实例
    llm = ChatOpenAI(
        model="qwen-max-0125",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=my_api_key,
        streaming=True
    )
    model_with_tools = llm.bind_tools(tools)
    workflow = StateGraph(MessagesState)
    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    app = workflow.compile()
    
    # example with a single tool call
    for chunk in app.stream(
        {"messages": [("human", "现在是北京时间几点？")]}, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

