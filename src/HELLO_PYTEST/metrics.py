'''Modules'''
from typing import List


def profit(revenue: List[float], costs: List[float]) -> float:
    '''Общая прибыль'''
    return sum(revenue) - sum(costs)


def margin(revenue: List[float], costs: List[float]) -> float:
    '''Маржинальность (отношение прибыли к выручке)'''
    return (sum(revenue) - sum(costs)) / sum(revenue)


def markup(revenue: List[float], costs: List[float]) -> float:
    '''Средняя наценка (отношение прибыли к затратам)'''
    return (sum(revenue) - sum(costs)) / sum(costs)
