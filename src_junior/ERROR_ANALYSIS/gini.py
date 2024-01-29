"""Fairness metric calculation"""
from typing import List
import numpy as np

from sklearn.metrics import log_loss

def gini(x: np.ndarray):
    """Gini coefficient calculation"""
    x = np.abs(x.copy())
    total = 0
    for xi in x:
        total += np.sum(np.abs(x - xi))
    return total / (2 * len(x)**2 * np.mean(x))

def fairness(residuals: np.ndarray) -> float:
    """Compute Gini fairness of array of values"""
    gini_coef = gini(residuals)
    return 1 - gini_coef

def best_prediction(
    y_true: np.ndarray, y_preds: List[np.ndarray], fairness_drop: float = 0.05
) -> int:
    """Find index of best model"""
    best_i = 0
    # base parameters
    base_loss = log_loss(y_true, y_preds[0])
    log_residuals = y_true*np.log(y_preds[0]) + (1-y_true)*np.log(1-y_preds[0])
    base_fairness = fairness(log_residuals)
    # seeking the best model
    for i, y_pred in enumerate(y_preds):
        # model loss term and Gini-fairness
        model_loss = log_loss(y_true, y_pred)
        log_residuals = y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)
        model_fairness = fairness(log_residuals)
        # if the model_loss is less and the fairness drop is not critical
        if (model_loss < base_loss and model_fairness > base_fairness * (1-fairness_drop)):
            # save the best model info
            base_loss = model_loss
            #base_fairness = model_fairness
            best_i = i

    return best_i
