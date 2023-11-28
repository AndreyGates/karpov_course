'''Modules'''
import numpy as np

def smape(y_true: np.array, y_pred: np.array) -> float:
    '''Calculates symmetric mean absolute precentage error'''
    abs_err = np.abs(y_true - y_pred) # the numerator

    devisor = np.abs(y_true) + np.abs(y_pred) # the denominator
    devisor[devisor == 0] = 1 # to avoid zero devision

    return np.mean(2 * abs_err / devisor)
    