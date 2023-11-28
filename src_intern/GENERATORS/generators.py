'''Modules'''
from typing import Tuple
import random

def username_generator(n: int, first_names=None, last_names=None) -> dict:
    '''
    Generates n id-fname-lname triples as dict
    '''
    first_names = first_names if first_names is not None else ['Robert', 'Emilia', 'Sam']
    last_names = last_names if last_names is not None else ['Downey', 'Swift', 'Dickens']

    # yield random names
    for i in range(1, 1+n):
        yield {'id': i,
               'first_name': random.choice(first_names),
               'last_name': random.choice(last_names)}

def data_generator(n: int) -> Tuple[int, int]:
    '''
    Generates n indexed random numbers
    (in range of 0 to 100)
    '''
    for x in range(n):
        y = random.randint(0, 100)
        yield x, y
