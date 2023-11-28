'''Modules'''
from typing import Tuple
import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean squared error loss function and gradient."""
    loss = np.mean(np.square(y_pred - y_true)) # quadratic loss
    grad = y_pred - y_true # gradient (vector taking all parameters)
    return loss, grad


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean absolute error loss function and gradient."""
    loss = np.mean(np.abs(y_pred - y_true)) # absolute loss
    grad = np.sign(y_pred - y_true) # gradient (vector taking all parameters)
    return loss, grad
