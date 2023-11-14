'''Modules'''
import numpy as np

def ltv_error(y_true: np.array, y_pred: np.array) -> float:
    '''Calculates Lifetime Value loss for all the clients'''
    # step 1: calculating squares for targets and predictions
    true_logs = np.square(y_true)
    pred_logs = np.square(y_pred)

    # step 2: calculating MAE
    error = np.mean(np.abs(true_logs - pred_logs))
    return error


#if __name__ == '__main__':
#    y_true = np.array([100, 50, 150])
#    y_pred = np.array([90, 40, 140])
#
#    print(ltv_error(y_true, y_pred))
