# Image Analysis Assistant

这是一个基于大语言模型的图像分析助手，可以对图片进行智能分析并支持多轮对话。

## 功能特点

- 基于 LangChain 框架实现的智能图像分析
- 支持多种图像处理和分析功能
- 可扩展的工具系统
- 配置化的模型和 API 管理

## 安装说明

1. 克隆项目到本地
2. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```
3. 在 `config/api_keys.json` 中配置必要的 API 密钥

## 目录结构

```
├── agents/                # Agent实现
│   ├── image_analysis.py  # 图片分析Agent
│   └── utils.py          # 工具函数
├── config/               # 配置文件
│   ├── api_keys.json    # API密钥
│   └── models.json      # 模型配置
├── data/                # 数据目录
│   ├── images/         # 图片存储
│   └── logs/           # 日志文件
├── tools/               # 工具类
│   └── image_tool.py   # 图片处理工具
├── tests/               # 测试用例
├── main.py              # 程序入口
└── requirements.txt     # 项目依赖
```

## 使用方法

1. 确保配置文件中包含必要的API密钥
2. 运行主程序：
   ```bash
   python main.py
   ```

## 开发指南

- 遵循PEP 8编码规范
- 新功能请添加对应的测试用例
- 保持代码文档的完整性

## 环境要求

- Python 3.8+
- 详细依赖见 requirements.txt
