'''Modules'''
from functools import reduce
from typing import List, Tuple


def sales_with_tax(sales: List[float], tax_rate: float, threshold: float = 300) -> List[float]:
    '''Applying taxes to high-volume sales'''
    taxed_sales = list(map(
                           lambda sale: (1+tax_rate)*sale,
                           filter(lambda sale: sale > threshold, sales)
                           ))
    return taxed_sales

def sum_sales(sales: List[float], threshold: float = 300) -> float:
    '''Summing sales satisfying the threshold'''
    summed_valid_sales = float(reduce(lambda x, y: x+y,
                                filter(lambda sale: sale > threshold, sales)))
    return summed_valid_sales

def average_age(ages: List[int], threshold: int = 30) -> float:
    '''Average age of those older than the threshold age'''
    filtered_age = list(filter(lambda age: age > threshold, ages))
    avg_age = reduce(lambda x, y: x+y, filtered_age) / len(filtered_age)
    return avg_age

def increased_prices(prices: List[float],
                     increase_rate: int = 0.2,
                     threshold: float = 300) -> List[float]:
    '''Filtering the increased prices'''
    filtered_prices = list(filter(lambda price: price > threshold,
                           map(lambda price: (1+increase_rate)*price, prices)))
    return filtered_prices

def weighted_sale_price(sales: List[Tuple[int, int]]) -> float:
    '''Calculating weighted average item price'''
    total_items = reduce(lambda x, y: x+y,
                         map(lambda sale: sale[1], sales))
    weighted_prices = reduce(lambda x, y: x+y,
                             map(lambda sale: sale[0]*sale[1]/total_items, sales))
    return weighted_prices
