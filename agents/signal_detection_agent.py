from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import dotenv_values
import os
import re
from datetime import datetime

# Import decision-making prompt template
from config.prompts import TRADE_SIGNAL_SYSTEM_PROMPT

from tools.new_stock_tools import StockAnalysis
from tools.agert_tools import LangGraphToolConverter
analysis_tool = StockAnalysis()

tools = LangGraphToolConverter().functions_to_tools([
    analysis_tool.get_stock_daily_data,
    analysis_tool.get_real_time_data,
    analysis_tool.calculate_ema,
    analysis_tool.calculate_sma,
    analysis_tool.calculate_macd,
    analysis_tool.calculate_rsi,
    analysis_tool.calculate_bollinger_bands,
    analysis_tool.calculate_atr,
    analysis_tool.calculate_obv,
    analysis_tool.calculate_kdj,
])

deepseek = False

class SignalDetectionResult(TypedDict):
    signal_type: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    technical_indicators: Dict[str, Any]
    risk_assessment: str
    timestamp: str

def detect_trading_signals(analysis_result: str, 
                          stock_code: str = "", 
                          stock_name: str = "", 
                          exchange: str = "",
                          current_position: int = 0,
                          current_cost: float = 0.0) -> SignalDetectionResult:
    """
    Based on a stock analysis result, determine if there are buy/sell signals
    using a large language model.
    
    Args:
        analysis_result: The detailed analysis text from trade_analysis_agent
        stock_code: The stock ticker code
        stock_name: The name of the stock
        exchange: The exchange where the stock is traded
        
    Returns:
        A SignalDetectionResult with the signal type and reasoning
    """
    config = dotenv_values("./.env")
    
    # Choose model based on available API keys
    if deepseek:
        my_api_key = config.get("DEEPSEEK_MODEL_API_KEY")
        model = ChatOpenAI(
            model="deepseek-reasoner",
            base_url="https://api.deepseek.com/v1",
            api_key=my_api_key,
            temperature=0.1
        )
    else:
        my_api_key = config.get("MODEL_API_KEY")
        model = ChatOpenAI(
            model="qwen3-235b-a22b",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",
            api_key=my_api_key,
            streaming=True
        )

    model = model.bind_tools(tools=tools)
    
    # Create a prompt that focuses on signal detection
    signal_prompt = ChatPromptTemplate.from_messages([
        ("system", TRADE_SIGNAL_SYSTEM_PROMPT),
        ("user", """
        请基于以下股票分析结果，判断当前是否存在买入或卖出信号：
        
        股票名称: {stock_name}
        股票代码: {stock_code}
        交易所: {exchange}
        当前日期: {current_date}
        当前持仓(股): {current_position}
        当前成本: {current_cost}
        
        初步分析结果:
        {analysis_result}
        
        请根据各种技术指标（如MACD、RSI、KDJ、布林带等）综合判断，并给出详细理由。
        """)
    ])
    
    # Get the current date
    current_date = datetime.now().strftime('%Y-%m-%d')

    if current_position == 0:
        current_cost = 0.0
    
    # Generate the signal detection result
    response = model.invoke(
        signal_prompt.format_messages(
            stock_name=stock_name,
            stock_code=stock_code,
            exchange=exchange,
            current_date=current_date,
            analysis_result=analysis_result,
            current_position=current_position,
            current_cost=current_cost
        )
    )
    
    # Parse the response to extract signal information
    # This assumes the model's response is structured in a way we can parse
    signal_content = response.content
    
    # Default values
    signal_type = "ERROR"
    confidence = 0
    reasoning = ""
    technical_indicators = {}
    risk_assessment = ""
    
    # Extract signal type using regex
    signal_match = re.search(r'信号类型[：:]\s*([买卖持观望]*|BUY|SELL|HOLD)', signal_content, re.IGNORECASE)
    if signal_match:
        signal_text = signal_match.group(1).upper()
        if "买" in signal_text or "BUY" in signal_text:
            signal_type = "BUY"
        elif "卖" in signal_text or "SELL" in signal_text:
            signal_type = "SELL"
        else:
            signal_type = "HOLD"
    
    # Extract confidence
    confidence_match = re.search(r'置信度[：:]\s*([\d.]+)', signal_content)
    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))
            if confidence > 1.0:  # Handle percentage format
                confidence = confidence / 100.0
        except ValueError:
            confidence = 0.5  # Default if parsing fails
    
    # Extract reasoning
    reasoning_match = re.search(r'理由[：:]([\s\S]*?)(?=技术指标[：:]|风险评估[：:]|$)', signal_content)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Extract risk assessment
    risk_match = re.search(r'风险评估[：:]([\s\S]*?)(?=\n\n|$)', signal_content)
    if risk_match:
        risk_assessment = risk_match.group(1).strip()
    
    # Create result dictionary
    result = SignalDetectionResult(
        signal_type=signal_type,
        confidence=confidence,
        reasoning=reasoning,
        technical_indicators=technical_indicators,  # This would need more parsing to extract specific indicators
        risk_assessment=risk_assessment,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    return result

def save_signal_result(signal_result: SignalDetectionResult, 
                      stock_code: str, 
                      stock_name: str, 
                      exchange: str) -> str:
    """
    Save the signal detection result as a markdown file.
    
    Args:
        signal_result: The detection result
        stock_code: Stock ticker symbol
        stock_name: Name of the stock
        exchange: Exchange where the stock is traded
        
    Returns:
        The path to the saved file
    """
    if not all([stock_code, stock_name, exchange]):
        raise ValueError("股票代码、名称和交易所不能为空")
    
    # Create save directory
    today = datetime.now().strftime('%Y%m%d%H')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    signals_dir = os.path.join(base_dir, "signals", f"{stock_name}_{stock_code}_{exchange}")
    os.makedirs(signals_dir, exist_ok=True)
    
    # Create filename
    filename = f"signal_{today}.md"
    filepath = os.path.join(signals_dir, filename)
    
    # Format content for markdown
    content = f"""# {stock_name} ({stock_code}) 交易信号

## 基本信息
- **股票名称**: {stock_name}
- **股票代码**: {stock_code}
- **交易所**: {exchange}
- **分析时间**: {signal_result['timestamp']}

## 信号详情
- **信号类型**: {signal_result['signal_type']}
- **置信度**: {signal_result['confidence']:.2f}

## 分析理由
{signal_result['reasoning']}

## 风险评估
{signal_result['risk_assessment']}
"""
    
    # Write to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    return filepath
