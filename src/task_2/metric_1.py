'''Modules'''
import numpy as np

def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    '''Takes observed and predicted turnover values. Calculates the total turnover loss'''

    # step 1: calculating 1 + 1/x for targets and predictions
    true_logs = np.log1p(1.0/y_true)
    pred_logs = np.log1p(1.0/y_pred)

    # step 2: calculating MSE
    error = np.mean(np.square(true_logs - pred_logs))
    return error
