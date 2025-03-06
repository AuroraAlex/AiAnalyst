"""
Main entry point for the Image Analysis Assistant.
"""

import os
from pathlib import Path
from agents.image_analysis import ImageAnalysisAgent
from tools.utils import load_config
from agents.indicators_analysis import IndicatorsAnalysisAgent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 配置代理
os.environ['http_proxy'] = 'http://127.0.0.1:2561'
os.environ['https_proxy'] = 'http://127.0.0.1:2561'

def main1():
    # Load configurations
    config_dir = Path(__file__).parent / "config"
    api_keys = load_config(config_dir / "api_keys.json")
    model_config = load_config(config_dir / "models.json")
    
    # Initialize agent
    agent = ImageAnalysisAgent(config={
        **api_keys,
        **model_config
    })
    agent.initialize()
    
    print("图像分析助手已启动（输入 'quit' 退出）")
    print("您可以直接聊天，或使用 '--image <图片路径>' 来分析图片")
    
    while True:
        user_input = input("\n用户: ").strip()
        
        if user_input.lower() == 'quit':
            break
            
        # 检查是否包含图片分析命令
        if user_input.startswith('--image'):
            try:
                _, image_path = user_input.split(maxsplit=1)
                image_path = Path(image_path)
                if not image_path.exists():
                    print(f"错误：找不到图片 {image_path}")
                    continue
                    
                query = input("请描述您想了解图片的什么内容: ")
                response = agent.chat(query, str(image_path))
            except ValueError:
                print("错误：请提供图片路径，例如：--image ./images/example.jpg")
                continue
        else:
            # 普通对话模式
            response = agent.chat(user_input)
            
        print("\n助手:", response)

def main2():
    # Load configurations
    config_dir = Path(__file__).parent / "config"
    api_keys = load_config(config_dir / "api_keys.json")
    model_config = load_config(config_dir / "models.json")
    
    # Initialize agent
    agent = IndicatorsAnalysisAgent(config={
        **api_keys,
        **model_config
    })


def main3():
    # 初始化Agent
    agent = IndicatorsAnalysisAgent()
    app = agent.creat_agent()
    messages = []
    #创建系统Message
    system_message = SystemMessage(content=[{
        "type": "text",
        "text": "你是一个股票分析师，我有一些股票指标需要你帮我分析，同时给出买入卖出的建议，你可以使用一些工具来帮助你分析"
    }])
    messages.append(system_message)
    #创建用户Message
    human_message = HumanMessage(content=[{
        "type": "text",
        "text": "我已经存储了商汤的近90天日k数据，请帮我分析一下最近90天的股票趋势，我想了解MACD和EMA指标带来的指示"
    }])
    messages.append(human_message)
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

def main():
    from tools import new_stock_tools
    stock = new_stock_tools.StockAnalysis()
    df= stock.get_stock_daily_data("81810", "XHKG", 360)
    RS_DF = stock.calculate_ema(df)
    js = stock.pd_json(RS_DF)
    print(RS_DF)

if __name__ == "__main__":
    main3()
    