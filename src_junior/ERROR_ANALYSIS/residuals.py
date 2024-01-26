"""Residuals calculation"""
import numpy as np

def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Residuals"""
    if y_true.shape != y_pred.shape:
        raise ValueError
    return y_true - y_pred

def squared_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Squared errors"""
    if y_true.shape != y_pred.shape:
        raise ValueError
    return (y_true - y_pred) ** 2

def logloss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """LogLoss terms"""
    if y_true.shape != y_pred.shape or\
       not all(proba > 0 and proba < 1 for proba in y_pred) or\
       not all(true in [0, 1] for true in np.unique(y_true)):
        raise ValueError

    return y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)

def ape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MAPE terms"""
    if y_true.shape != y_pred.shape or\
       not all(true >= 0 for true in y_true) or\
       not all(proba >= 0 for proba in y_pred):
        raise ValueError

    return (y_true-y_pred) / (y_true+0.00000000001) # smoothing factor

def quantile_loss(
    y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.01
) -> np.ndarray:
    """Quantile loss terms"""
    if y_true.shape != y_pred.shape:
        raise ValueError

    errors = y_true - y_pred
    loss = np.maximum(q * errors, (q-1) * errors)
    return loss
