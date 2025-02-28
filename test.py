import requests

 

url = f"https://tsanghi.com/api/fin/stock/XHKG/list?token=aa0c30ad51024a5885045206bbd0ae24"

data = requests.get(url).json()

print(data)
with open("xhkg_stock.json", "w",encoding='utf-8') as f:
    f.write(str(data))