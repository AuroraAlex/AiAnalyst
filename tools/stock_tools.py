import requests
import json
import os
import utils
from typing import List, Dict, Any
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
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

    def get_stock_daily_data(self,stock_code, exchange_code):
        """
        查询指定股票过去一个月的日线数据

        :param stock_code: 股票代码 (例如 "AAPL")
        :param exchange_code: 交易所代码 (例如 "NASDAQ")
        :return: 返回日线数据的JSON对象
        """
        # 获取当前日期和一个月前的日期
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")

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

    def plot_stock_daily(self, stock_code: str, exchange_code: str, save_path: str = None) -> None:
        """
        绘制专业股票K线图并保存

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

        # 设置K线图样式
        mc = mpf.make_marketcolors(
            up='red',
            down='green',
            edge='inherit',
            volume='in',
            wick={'up':'red', 'down':'green'}
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='--',
            y_on_right=True,
            base_mpl_style='seaborn',
            rc=style_override
        )

        # 创建图表配置
        kwargs = dict(
            type='candle',
            volume=True,
            figratio=(16, 9),
            figscale=1.2,
            title=f'{stock_code} 股票K线图',
            ylabel='价格',
            ylabel_lower='成交量',
            style=s,
            tight_layout=True,
            update_width_config=dict(
                candle_linewidth=0.8,
                candle_width=0.8,
                volume_linewidth=0.8,
                volume_width=0.8
            ),
            scale_padding={'left': 0.5, 'right': 1, 'top': 2, 'bottom': 1}
        )

        # 如果需要保存图片
        if save_path:
            kwargs['savefig'] = dict(
                fname=save_path,
                dpi=300,
                bbox_inches='tight'
            )

        # 绘制图表
        mpf.plot(df, **kwargs)
        
        if not save_path:
            plt.show()
        else:
            print(f"图表已保存至: {save_path}")

if __name__ == "__main__":
    config_dir = Path(__file__).parent.parent / "config"
    api_keys = utils.load_config(config_dir / "api_keys.json")
    api_config = utils.load_config(config_dir / "models.json")

    stock_query = StockQuery(config={
        **api_keys,
        **api_config
    })
    
    # 获取特斯拉股票数据并绘制图表
    stock_query.plot_stock_daily("TSLA", "XNAS", "tesla_stock.png")
    
    # 获取可口可乐股票数据并绘制图表
    stock_query.plot_stock_daily("KO", "XNYS", "coca_cola_stock.png")