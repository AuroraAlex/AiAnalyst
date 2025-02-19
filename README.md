# Image Analysis Assistant

这是一个基于大语言模型的图像分析助手，可以对图片进行智能分析并支持多轮对话。

## 功能特点

- 支持图像分析和描述
- 支持多轮对话
- 支持处理外部图片
- 结果保存和格式化输出
- 可选的 JSON 响应保存

## 配置说明

在使用前，需要创建 `config.json` 文件，包含以下配置：

```json
{
    "api_key": "你的API密钥",
    "image_folder": "images",
    "output_folder": "outputs",
    "base_url": "API基础URL",
    "model_config": {
        "model": "模型名称"
    },
    "save_json": false
}
```

## 目录结构

```
├── config.json          # 配置文件
├── deepseek_openai_style.py  # 主程序
├── images/             # 图片目录
└── outputs/           # 输出目录
```

## 使用方法

1. 安装依赖：
```bash
pip install openai
```

2. 配置 config.json

3. 运行程序：
```python
from deepseek_openai_style import ImageProcessor, ConfigManager

config = ConfigManager()
processor = ImageProcessor(config)

# 处理图片
processor.process_image("example.png", "图中描绘的是什么景象？")

# 继续对话
processor.continue_conversation("请详细分析一下图中的内容。")
```

## 注意事项

- 请确保 config.json 中的 API 密钥配置正确
- 图片文件请放在 images 目录下
- 分析结果将保存在 outputs 目录中