'''Modules'''
import numpy as np

def mse(y: np.ndarray) -> float:
    """Compute the mean squared error of a vector."""
    return np.mean((np.square(np.mean(y) - y)))


def weighted_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute the weighted mean squared error of two vectors."""
    n_left, n_right = y_left.shape[0], y_right.shape[0]
    mse_left, mse_right = mse(y_left), mse(y_right)
    # MSEs weighted sum
    return (n_left*mse_left + n_right*mse_right) / (n_left + n_right)
