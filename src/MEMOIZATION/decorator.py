'''Modules'''
from typing import Callable

def memoize(func: Callable) -> Callable:
    """
    Memoize function
    """
    cache = {}

    def wrapper(*args, **kwargs) -> object:
        """
        The wrapper collect all the agrument values 
        and the return value of a function
        """
        # gathering positional and keyword agrs
        # (priorly resolving 'unhashable type: list' issue)
        key = str(args) + str(kwargs)

        # if the current arguments are new,
        # calculate the function cache them
        if key not in cache:
            cache[key] = func(*args, **kwargs)

        return cache[key]
    return wrapper
