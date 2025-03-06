import requests
import json
from dotenv import load_dotenv
import os
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
from tools import utils


class StockAnalysis:
    def __init__(self):
        # 加载 .env 文件
        load_dotenv()
        self.data_api_key = os.getenv("STOCK_DATA_API_KEY")
        self.data_base_url = os.getenv("STOCK_DATA_BASE_URL")

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
        url = f"{self.data_base_url}api/fin/stock/{exchange_code}/daily?token={self.data_api_key}&ticker={stock_code}&start_date={start_date}&end_date={end_date}&order=1"

        # 检查请求是否成功
        try:
            # Get the response first
            response = requests.get(url)
            # Check HTTP status code
            response.raise_for_status()
            # Parse JSON data
            data = response.json()
            
            if data.get("code") == 200:
                df = pd.DataFrame(data.get("data"))
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                
                return df
            else:
                print(f"API Error: {data.get('code')}")
                return None
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {str(e)}")
            return None
        
    
    def calculate_sma(self, df: pd.DataFrame, period: int = 20):
        """
        计算简单移动平均线（SMA）

        :param df: 包含股票数据的Pandas DataFrame
        :param period: SMA的计算周期
        :return: 包含SMA数据的Pandas DataFrame
        """
        # 计算SMA指标
        sma = ta.sma(df["close"], length=period)
        return sma.round(2)
    
    def calculate_ema(self, df: pd.DataFrame, period: int = 20):
        """
        计算指数移动平均线（EMA）

        :param df: 包含股票数据的Pandas DataFrame
        :param period: EMA的计算周期
        :return: 包含EMA数据的Pandas DataFrame
        """
        # 计算EMA指标
        ema = ta.ema(df["close"], length=period)
        return ema.round(2)
    
    def calculate_macd(self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        计算移动平均收敛-散度指标（MACD）

        :param df: 包含股票数据的Pandas DataFrame
        :param fast_period: MACD的快速线计算周期
        :param slow_period: MACD的慢速线计算周期
        :param signal_period: MACD的信号线计算周期
        :return: 包含MACD数据的Pandas DataFrame,pd.DataFrame: macd, histogram, signal columns
        """
        # 计算MACD指标
        df = df.sort_index(ascending=True)
        macd = ta.macd(df["close"], fast=fast_period, slow=slow_period, signal=signal_period)
        macd.columns = ['macd', 'histogram', 'signal']
        return macd.round(2)
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14):
        """
        计算相对强弱指标（RSI）

        :param df: 包含股票数据的Pandas DataFrame
        :param period: RSI的计算周期
        :return: 包含RSI数据的Pandas DataFrame
        """
        # 计算RSI指标
        rsi = ta.rsi(df["close"], length=period)
        return rsi.round(2)
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: int = 2):
        """
        计算布林带指标

        :param df: 包含股票数据的Pandas DataFrame
        :param period: 布林带的计算周期
        :param std_dev: 布林带的标准差
        :return: 包含布林带数据的Pandas DataFrame,pd.DataFrame: lower, mid, upper, bandwidth, and percent columns.
        """
        # 计算布林带指标
        bbol= ta.bbands(df["close"], length=period, std=std_dev)
        bbol.columns = ['lower', 'mid', 'upper', 'bandwidth', 'percent']
        return bbol.round(2)
    

    def calculate_kdj(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        """
        计算随机指标（KDJ）

        :param df: 包含股票数据的Pandas DataFrame
        :param k_period: KDJ的K线计算周期
        :param d_period: KDJ的D线计算周期
        :return: 包含KDJ数据的Pandas DataFrame
        """
        # 计算KDJ指标
        kdj = ta.kdj(high=df['high'], low=df['low'], close=df['close'], fillna=float('nan'),length=k_period,signal=d_period)
        kdj.columns = ['k', 'd', 'j']
        return kdj.round(2)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14):
        """
        计算真实波幅指标（ATR）

        :param df: 包含股票数据的Pandas DataFrame
        :param period: ATR的计算周期
        :return: 包含ATR数据的Pandas DataFrame
        """
        # 计算ATR指标
        atr = ta.atr(df['high'], df['low'], df['close'], length=period)
        return atr.round(2)
    
    def calculate_obv(self, df: pd.DataFrame):
        """
        计算能量潮指标（OBV）

        :param df: 包含股票数据的Pandas DataFrame
        :return: 包含OBV数据的Pandas DataFrame
        """
        # 计算OBV指标
        obv = ta.obv(df['close'], df['volume'])
        return obv.round(2)
    
    def calculate_all(self):
        """
        计算所有技术指标

        :return: 包含所有技术指标的Pandas DataFrame
        """
        pass

    @staticmethod
    def pd_json(df: pd.DataFrame):
        """
        将Pandas DataFrame转换为JSON格式

        :param df: 包含股票数据的Pandas DataFrame
        :return: 包含股票数据的JSON对象
        """

        return df.to_json(orient='split',index=False )