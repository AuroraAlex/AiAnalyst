import requests
import json
import os
import utils
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import matplotlib as mpl
from matplotlib.font_manager import findfont, FontProperties

# 查找系统中可用的中文字体
def find_chinese_font():
    fonts = ['SimHei', 'Microsoft YaHei', 'SimSun']
    for font_name in fonts:
        try:
            if findfont(FontProperties(family=font_name)) is not None:
                return font_name
        except:
            continue
    return 'SimHei'  # 默认使用黑体

def configure_plot_style():
    """Configure global plot style settings."""
    chinese_font = find_chinese_font()
    plt.rcParams['font.sans-serif'] = [chinese_font]
    plt.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['font.family'] = ['sans-serif']
    
    return {
        'font.family': 'sans-serif',
        'font.sans-serif': [chinese_font],
        'axes.unicode_minus': False,
    }

class StockQuery:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._initialize_api()
        self.style_override = configure_plot_style()
    def _initialize_api(self):
        """Initialize the StockQuery API with configuration."""
        api_keys = self.config.get("api_keys", {})
        stock_config = self.config.get("others", {}).get("stock_api", {})
        self.base_url = stock_config.get("canghai_base_url")
        self.api_key = api_keys.get("canghai_api_key")

    def get_stock_daily_data(self,stock_code, exchange_code,days=90):
        """
        查询指定股票过去一个月的日线数据

        :param stock_code: 股票代码 (例如 "AAPL")
        :param exchange_code: 交易所代码 (例如 "NASDAQ")
        :return: 返回日线数据的JSON对象
        """
        # 获取当前日期和一个月前的日期
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days)).strftime("%Y-%m-%d")

        # 发送请求到沧海数据API
        # 直接构造完整URL
        url = f"{self.base_url}api/fin/stock/{exchange_code}/daily?token={self.api_key}&ticker={stock_code}&start_date={start_date}&end_date={end_date}&order=1"

        # 检查请求是否成功
        try:
            # Get the response first
            response = requests.get(url)
            # Check HTTP status code
            response.raise_for_status()
            # Parse JSON data
            data = response.json()
            
            if data.get("code") == 200:
                return data.get("data")
            else:
                print(f"API Error: {data.get('code')}")
                return None
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {str(e)}")
            return None

    def format_volume(self, x, p):
        """
        格式化成交量显示，添加K/M/B单位
        """
        if x >= 1e9:
            return f'{x/1e9:.1f}B'
        elif x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.1f}K'
        return f'{x:.0f}'

    def plot_stock_daily(self, stock_code: str, exchange_code: str, save_path: str = None) -> None:
        """
        绘制专业股票K线图并保存，成交量显示在独立子图中，成交量颜色跟随K线涨跌

        :param stock_code: 股票代码
        :param exchange_code: 交易所代码
        :param save_path: 图片保存路径，如果为None则显示图片
        """
        data = self.get_stock_daily_data(stock_code, exchange_code)
        if not data:
            print("无法获取股票数据")
            return

        # 将数据转换为pandas DataFrame格式
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # 重命名列以符合mplfinance要求
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        # 计算涨跌
        df['price_change'] = df['Close'] - df['Close'].shift(1)
        
        # 设置K线图样式
        mc = mpf.make_marketcolors(
            up='#ff3333',          # 上涨为鲜红色
            down='#00aa00',        # 下跌为深绿色
            edge='inherit',
            wick='inherit',
            volume={'up':'#ff3333', 'down':'#00aa00'},  # 成交量用略浅的红绿色
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='-',  # 实线网格
            gridaxis='both',
            y_on_right=True,
            base_mpl_style='seaborn',
            rc={
                **self.style_override,
                'grid.alpha': 0.3,  # 网格线透明度
                'grid.linewidth': 0.5,  # 网格线宽度
                'grid.color': '#DDDDDD',  # 设置网格线颜色为淡灰色
                'axes.facecolor': '#FFFFFF',  # 设置图表背景为白色
                'savefig.facecolor': '#FFFFFF',  # 设置保存图片时的背景为白色
                'figure.facecolor': '#FFFFFF',  # 设置图形背景为白色
            }
        )

        # 创建图表和子图
        fig, axlist = mpf.plot(df, type='candle', 
                             volume=True,
                             figsize=(16, 9),
                            #  title=f'{stock_code} 股票K线图',
                             ylabel='价格',
                             ylabel_lower='成交量 (股)',
                             style=s,
                             volume_panel=1,
                             panel_ratios=(3.5,1),
                             returnfig=True,
                             tight_layout=True,
                             scale_padding={'left': 0.1, 'right': 0.1},
                             warn_too_much_data=100000,
                             show_nontrading=False)  # 不显示非交易日期的网格线
        
        # 获取主图和成交量子图
        ax1, ax2 = axlist[0], axlist[2]
        
        # 设置成交量y轴格式化
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(self.format_volume))
        ax2.set_ylabel('成交量 (股)', labelpad=3)  # 进一步减少标签和轴的间距

        # 调整坐标轴和标题
        for ax in [ax1, ax2]:
            ax.tick_params(axis='both', which='major', labelsize=9)  # 调整刻度标签大小
            ax.tick_params(axis='x', which='major', rotation=15)  # 稍微旋转x轴标签
            # 调整y轴标签位置
            ax.yaxis.set_label_coords(-0.04, 0.5)  # 微调y轴标签位置
        
        # 移动标题位置
        ax1.set_title(f'{stock_code} 股票K线图', pad=3, y=1.01)  # 微调标题位置

        # 调整图表布局，减少留白
        plt.subplots_adjust(left=0.06, right=0.94, hspace=0.08, top=0.96)  # 进一步优化整体布局

        # 如果需要保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)  # 减少保存时的边距
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()
            
        plt.close()

    @staticmethod
    def json_data_to_pd(data):
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df

    def calculate_technical_indicators(self, df: pd.DataFrame, 
                                params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        计算各种技术指标

        Args:
            df: 包含 OHLCV 数据的 DataFrame
            params: 包含各个指标参数的字典，可选，格式如：
                {
                    'sma': [5, 10, 20],  # SMA周期
                    'ema': [12, 26],     # EMA周期
                    'macd': {'fast': 12, 'slow': 26, 'signal': 9},  # MACD参数
                    'rsi': [14],         # RSI周期
                    'bbands': {'length': 20, 'std': 2},  # 布林带参数
                    'volume_sma': [5, 10, 20],  # 成交量SMA周期
                }
        
        Returns:
            包含各类技术指标的字典
        """
        if params is None:
            params = {
                'sma': [5, 10, 20],
                'ema': [12, 26],
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'rsi': [14],
                'bbands': {'length': 20, 'std': 2},
                'volume_sma': [5, 10, 20]
            }
        
        indicators = {}
        
        # 移动平均线 (SMA)
        for period in params.get('sma', []):
            indicators[f'sma_{period}'] = df.ta.sma(length=period)
        
        # 指数移动平均线 (EMA)
        for period in params.get('ema', []):
            indicators[f'ema_{period}'] = df.ta.ema(length=period)
        
        # MACD
        macd_params = params.get('macd', {'fast': 12, 'slow': 26, 'signal': 9})
        macd = df.ta.macd(
            fast=macd_params['fast'],
            slow=macd_params['slow'],
            signal=macd_params['signal']
        )
        # 获取 MACD 指标的列名
        macd_cols = macd.columns
        indicators['macd'] = {
            'macd_line': macd[macd_cols[0]],  # MACD 线
            'signal_line': macd[macd_cols[1]], # Signal 线
            'histogram': macd[macd_cols[2]]    # Histogram
        }
        
        # RSI
        for period in params.get('rsi', []):
            indicators[f'rsi_{period}'] = df.ta.rsi(length=period)
        
        # 布林带
        bbands_params = params.get('bbands', {'length': 20, 'std': 2})
        bbands = df.ta.bbands(
            length=bbands_params['length'],
            std=bbands_params['std']
        )
        bbands_cols = bbands.columns
        indicators['bollinger_bands'] = {
            'upper': bbands[bbands_cols[0]],   # 上轨
            'middle': bbands[bbands_cols[1]],  # 中轨
            'lower': bbands[bbands_cols[2]]    # 下轨
        }
        
        # 成交量移动平均
        for period in params.get('volume_sma', []):
            indicators[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        
        # KDJ
        stoch = df.ta.stoch(k=14, d=3, smooth_k=3)
        stoch_cols = stoch.columns
        indicators['kdj'] = {
            'k': stoch[stoch_cols[0]],
            'd': stoch[stoch_cols[1]]
        }
        
        # ATR - Average True Range
        indicators['atr'] = df.ta.atr(length=14)
        
        # OBV - On Balance Volume
        indicators['obv'] = df.ta.obv()
        
        return indicators

    def analyze_trading_data(self, trading_data: List[Dict]) -> Dict[str, Any]:
        """
        分析交易数据并计算技术指标

        Args:
            trading_data: 包含交易数据的列表，每个字典应包含日期、开盘价、最高价、最低价、收盘价和成交量

        Returns:
            包含分析结果和技术指标的字典
        """
        df = pd.DataFrame(trading_data)
        
        # 确保数据格式正确
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("交易数据缺少必要的列：date, open, high, low, close, volume")

        # 设置日期索引
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # 计算基本统计数据
        stats = {
            'period_return': ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100),
            'volatility': df['close'].pct_change().std() * np.sqrt(252) * 100,  # 年化波动率
            'highest_price': df['high'].max(),
            'lowest_price': df['low'].min(),
            'avg_volume': df['volume'].mean(),
        }
        
        # 计算技术指标
        indicators = self.calculate_technical_indicators(df)
        
        return {
            'statistics': stats,
            'technical_indicators': indicators
        }

if __name__ == "__main__":
    config_dir = Path(__file__).parent.parent / "config"
    api_keys = utils.load_config(config_dir / "api_keys.json")
    api_config = utils.load_config(config_dir / "models.json")

    stock_query = StockQuery(config={
        **api_keys,
        **api_config
    })

    data = stock_query.get_stock_daily_data("AAPL", "XNAS", 120)
    df = stock_query.json_data_to_pd(data)
    analyze_trading_data = stock_query.analyze_trading_data(data)
    print(analyze_trading_data)
    # 获取特斯拉股票数据并绘制图表
    # stock_query.plot_stock_daily("TSLA", "XNAS", "tesla_stock.png")
    
    # # 获取可口可乐股票数据并绘制图表
    # stock_query.plot_stock_daily("KO", "XNYS", "coca_cola_stock.png")