"""
Main entry point for the Image Analysis Assistant.
"""

import os
from pathlib import Path
from agents.image_analysis import ImageAnalysisAgent
from tools.utils import load_config

def main():
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

if __name__ == "__main__":
    main()