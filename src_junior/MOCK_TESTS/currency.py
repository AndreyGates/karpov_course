"""Main program"""
import requests

def get_exchange_rate(base_currency, target_currency):
    """Server request for currency exchange rate info"""
    response = requests.get(f'https://www.xe.com/currencyconverter/convert/?Amount=1&From={base_currency}&To={target_currency}',
                            timeout=10)
    if response.status_code == 200:
        return response.json()["rate"]
    else:
        return None
