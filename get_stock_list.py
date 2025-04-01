import requests

def get_stock_list(exchange_code: str = "XNAS") -> None:
    """
    Fetches the stock list from the API and saves it to a JSON file.
    """
    # URL for the API endpoint
    url = f"https://tsanghi.com/api/fin/stock/"+exchange_code+"/list?token=aa0c30ad51024a5885045206bbd0ae24"

    data = requests.get(url).json()

    with open(exchange_code+"_stock.json", "w", encoding="utf-8") as f:
        f.write(str(data))


if __name__ == "__main__":
    get_stock_list()