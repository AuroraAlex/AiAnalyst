from agents.indicators_analysis import IndicatorsAnalysisAgent
from langchain_core.messages import SystemMessage, HumanMessage
import asyncio

async def main():
    # 初始化Agent
    print("初始化IndicatorsAnalysisAgent...")
    agent = IndicatorsAnalysisAgent()
    app = agent.creat_agent()
    
    messages = []
    # 创建系统Message
    system_message = SystemMessage(content=[{
        "type": "text",
        "text": "你是一个股票分析师，我有一些股票指标需要你帮我分析，同时给出买入卖出的建议，你可以使用一些工具来帮助你分析"
    }])
    messages.append(system_message)
    
    # 创建用户Message
    human_message = HumanMessage(content=[{
        "type": "text",
        "text": "我已经存储了商汤的近90天日k数据，请帮我分析一下最近90天的股票趋势，我想请你结合技术指标分析中期、短期的买入卖出建议，以及在何时、什么价位进行买入卖出"
    }])
    messages.append(human_message)
    
    print("开始调用Agent...")
    # 流式输出
    input = {"messages": messages}
    try:
        async for event in app.astream_events(input):
            if event["event"] == "on_chat_model_end":
                print(event["data"]["output"].content, end="", flush=True)
            if event["event"] == "on_tool_end":
                print(event["data"]["output"].content, end="", flush=True)
    except:
        print("发生异常，退出程序")
if __name__ == "__main__":
    asyncio.run(main())
