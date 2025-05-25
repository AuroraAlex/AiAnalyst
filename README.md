# AI Analyst Assistant

这是一个多功能AI分析助手，集成了图像分析、股票数据分析等功能，支持多轮对话的智能分析系统。

## 功能特点

- 基于 LangChain 框架实现的智能分析系统
- 支持图像处理和分析
- 集成股票数据查询和分析功能
  - 支持港股、美股、A股等多市场数据
  - 提供股票基础数据和指标分析
  - **新功能**: 基于技术分析自动检测股票买入/卖出信号
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
├── agents/                     # Agent实现
│   ├── image_analysis.py       # 图像分析Agent
│   ├── plan_agent.py          # 计划执行Agent
│   ├── trade_analysis_agent.py # 交易分析Agent
│   ├── signal_detection_agent.py # 买卖信号检测Agent
│   └── web_search.py          # 网络搜索Agent
├── config/                    # 配置文件
│   └── models.json           # 模型配置
├── tools/                    # 工具类
│   ├── image_tool.py        # 图片处理工具
│   ├── stock_tools.py       # 股票数据工具
│   ├── new_stock_tools.py   # 新版股票分析工具
│   └── utils.py             # 通用工具函数
├── graphs/                   # 图表生成
├── results/                  # 分析结果存储
├── signals/                  # 交易信号结果存储
├── tests/                    # 测试用例
├── main.py                  # 主程序入口
├── run_signal_detection.py  # 交易信号检测脚本
└── requirements.txt         # 项目依赖
```

## 快速开始

1. 克隆项目并安装依赖:
   ```bash
   git clone [项目地址]
   cd AiAnalyst
   pip install -r requirements.txt
   ```

2. 配置 API 密钥:
   在项目根目录创建 `.env` 文件，并设置相应的 API 密钥

3. 运行股票分析:
   ```bash
   python main.py --question "分析股票 小米集团 代码: 1810 exchange_code: XHKG"
   ```

4. 运行交易信号检测:
   ```bash
   # 使用已有的分析结果
   python run_signal_detection.py --question "分析股票 小米集团 代码: 1810 exchange_code: XHKG"
   
   # 强制重新运行分析
   python run_signal_detection.py --question "分析股票 小米集团 代码: 1810 exchange_code: XHKG" --run-analysis
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

### 4. 交易信号检测
- 基于技术分析的自动化交易信号生成
- 支持买入/卖出信号的智能检测
- 可视化信号分析结果

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
