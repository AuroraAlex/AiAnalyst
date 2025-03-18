from agents.function_calling_convert import FunctionCallingAgent
from langchain_core.messages import SystemMessage, HumanMessage
import asyncio
import traceback
import sys

async def main():
    # 初始化Agent
    print("初始化FunctionCallingAgent...")
    agent = FunctionCallingAgent()
    app = agent.creat_agent()
    
    messages = []
    # 创建系统Message
    system_message = SystemMessage(content=[{
        "type": "text",
        "text": "你是一个function calling转发工具，需要你来帮助没有实现function calling功能的模型调用function calling功能，你需要分析用户的输入，然后调用function calling工具来处理"
    }])
    messages.append(system_message)
    
    # 创建用户Message
    human_message = HumanMessage(content=[{
        "type": "text",
        "text": """
        根据分析需求，我将按以下顺序调用相关工具获取数据：
        1. **基础数据准备**
        - 执行 `get_120_days_data()` 获取近期价格走势基础数据

        2. **中期趋势判断**
        - 执行 `get_stock_macd(fast_period=12, slow_period=26, signal_period=9)` 验证MACD金叉状态
        - 执行 `get_stock_sma(period=30)` 和 `get_stock_sma(period=60)` 获取双均线数据
        - 执行 `get_stock_bollinger_bands(period=20, std_dev=2)` 获取布林通道

        3. **短期信号确认**
        - 执行 `get_stock_rsi(period=14)` 检测超卖区域
        - 执行 `get_stock_kdj(k_period=14, d_period=3)` 获取随机指标
        - 执行 `get_stock_obv()` 验证量价配合

        4. **风险控制参数**
        - 执行 `get_stock_atr(period=14)` 获取波动率数据
        - 计算20日成交量均值（需要基础数据中的成交量序列）

        正在执行工具调用，请稍候...（您需要先返回上述函数调用结果，我将基于实际数据继续完成分析）"""
    }])
    messages.append(human_message)
    
    print("开始调用Agent...")
    # 流式输出
    input = {"messages": messages}
    try:
        async for event in app.astream_events(input):
            if event["event"] == "on_tool_end":
                print(event["data"]["output"].content, end="", flush=True)
    except Exception as e:
        print(f"\n发生异常: {type(e).__name__}: {str(e)}")
        print("异常详情:")
        traceback.print_exc(file=sys.stdout)
        print("\n退出程序")

if __name__ == "__main__":
    asyncio.run(main())
