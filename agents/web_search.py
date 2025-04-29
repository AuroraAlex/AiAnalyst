from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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
config = dotenv_values("./.env")
my_api_key = config.get("MODEL_API_KEY")

# 配置代理
os.environ['http_proxy'] = 'http://127.0.0.1:6789'
os.environ['https_proxy'] = 'http://127.0.0.1:6789'

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
        return summary


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
        model="qwq-plus-latest",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=my_api_key,
        streaming=True
    )
    # llm = ChatOpenAI(
    #     model="deepseek-chat",
    #     base_url="https://api.deepseek.com",
    #     api_key="sk-e93fcab2961d421aa3347db4b7d7e547",
    #     streaming=True
    # )
    model_with_tools = llm.bind_tools(tools)
    workflow = StateGraph(MessagesState)
    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    app = workflow.compile()
    messages = []
    messages.append(SystemMessage(content=[{"type": "text", "text": "我提供给你一个web搜索工具搜索问题，搜索一次后根据我返回的结果选择你认为最好的一个，找到其中的链接，并再次调用工具进行搜索,此时除function calling调用不返回其他内容；最后查看链接的内容总结后返回我需要的内容"}]))
    messages.append(HumanMessage(content=[{"type": "text", "text": "如何使用Python搜索网络内容？"}]))
    #流式输出
    for message in app.invoke({"messages": messages}).get("messages", []):
        # 安全地处理消息内容，适应不同格式
        if hasattr(message, 'content'):
            if isinstance(message.content, list) and len(message.content) > 0:
                if isinstance(message.content[0], dict) and "text" in message.content[0]:
                    print(message.content[0]["text"])
                else:
                    print(str(message.content[0]))
            else:
                print(str(message.content))
        else:
            print(str(message))


