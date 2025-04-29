# AI Analyst Assistant

这是一个多功能AI分析助手，集成了图像分析、股票数据分析等功能，支持多轮对话的智能分析系统。

## 功能特点

- 基于 LangChain 框架实现的智能分析系统
- 支持图像处理和分析
- 集成股票数据查询和分析功能
  - 支持港股、美股、A股等多市场数据
  - 提供股票基础数据和指标分析
- 可扩展的工具系统
- 配置化的模型和 API 管理
- 支持自定义分析流程和计划

## 安装说明

1. 克隆项目到本地
2. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```
3. 在 `.env` 中配置必要的 API 密钥

## 目录结构

```
├── agents/                # Agent实现
│   ├── image_analysis.py  # 图像分析Agent
│   ├── plan_agent.py     # 计划执行Agent
│   └── web_search.py     # 网络搜索Agent
├── config/               # 配置文件
│   └── models.json      # 模型配置
├── tools/               # 工具类
│   ├── image_tool.py   # 图片处理工具
│   ├── stock_tools.py  # 股票数据工具
│   ├── new_stock_tools.py # 新版股票分析工具
│   └── utils.py        # 通用工具函数
├── graphs/              # 图表生成
├── tests/               # 测试用例
├── main.py             # 主程序入口
└── requirements.txt    # 项目依赖
```

## 快速开始

1. 克隆项目并安装依赖:
   ```bash
   git clone [项目地址]
   cd AiAnalyst
   pip install -r requirements.txt
   ```

2. 目前支持使用的功能
- 股票技术面分析
```bash
streamlit run stock_search.py
```

## 主要功能说明

### 1. 股票数据分析
- 支持多个市场的股票数据查询（港股、美股、A股）
- 提供股票基础数据和技术指标分析
- 可视化数据展示和分析结果

### 2. 图像分析
- 智能图像识别和分析
- 支持多轮对话式分析
- 可扩展的图像处理工具

### 3. 智能计划执行
- 支持自定义分析流程
- 智能任务规划和执行
- 灵活的工具调用系统

## 环境要求

- Python 3.8+
- LangChain 框架
- 推荐使用虚拟环境进行开发
- 详细依赖见 requirements.txt

## 注意事项

- 使用前请确保相关 API 配置正确
- 股票数据分析功能需要联网使用
- 建议在使用前查看各功能模块的示例代码
- 如遇问题请查看 tests 目录下的测试用例作为参考
