import requests
import json
import os
import utils
from typing import List, Dict, Any
from pathlib import Path
import datetime

class StockQuery:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._initialize_api()
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
        
if __name__ == "__main__":
    config_dir = Path(__file__).parent.parent / "config"
    api_keys = utils.load_config(config_dir / "api_keys.json")
    api_config = utils.load_config(config_dir / "models.json")

    stock_query = StockQuery(config={
        **api_keys,
        **api_config
    })
    data = stock_query.get_stock_daily_data("TSLA", "XNAS")
    if data:
        print(data)
    else:
        print("Failed to fetch stock data.")