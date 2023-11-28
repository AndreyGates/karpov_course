'''Modules'''
from typing import List
import numpy as np


def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
    """Discounted Cumulative Gain

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    relevance = np.array(relevance[:k])
    if method == 'standard':
        numerator = relevance

    elif method == 'industry':
        numerator = np.exp2(relevance) - 1.0

    else: # for any other value
        raise ValueError

    # calculating the common denominator and summing over all the fractions
    denominator = np.log2(1.0 + np.arange(1, len(relevance)+1))
    score = np.sum(np.divide(numerator, denominator))
    return score
