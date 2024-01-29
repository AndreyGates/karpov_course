"""Best-worse cases analysis"""
from typing import Optional

import numpy as np
import pandas as pd
import residuals


def best_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    top_k: int = 10,
    mask: Optional[pd.Series] = None,
    func: Optional[str] = None,
) -> dict:
    """Return top-k best cases according to the given function"""
    # masking
    if mask is not None:
        X_test = X_test[mask]
        y_test = y_test[mask]
        y_pred = y_pred[mask]

    # residuals calculation (saving absolute values)
    if func is None: # if no specific func, apply subtraction
        func = 'residuals'
    resid_func = getattr(residuals, func)
    resid = np.abs(resid_func(y_test, y_pred))

    # sorting data (choosing the best K residuals)
    sorted_idx = np.argsort(resid)[:top_k]
    X_test = X_test.iloc[sorted_idx]
    y_test = y_test.iloc[sorted_idx]
    y_pred = y_pred.iloc[sorted_idx]
    resid = pd.Series(resid).iloc[sorted_idx]

    # outputting the result
    result = {
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "resid": resid,
    }
    return result


def worst_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    top_k: int = 10,
    mask: Optional[pd.Series] = None,
    func: Optional[str] = None,
) -> dict:
    """Return top-k worst cases according to the given function"""
    # masking
    if mask is not None:
        X_test = X_test[mask]
        y_test = y_test[mask]
        y_pred = y_pred[mask]

    # residuals calculation (saving absolute values)
    if func is None: # if no specific func, apply subtraction
        func = 'residuals'
    resid_func = getattr(residuals, func)
    resid = np.abs(resid_func(y_test, y_pred))

    # sorting data (choosing the worst K residuals)
    sorted_idx = np.argsort(resid)[::-1][:top_k]
    X_test = X_test.iloc[sorted_idx]
    y_test = y_test.iloc[sorted_idx]
    y_pred = y_pred.iloc[sorted_idx]
    resid = pd.Series(resid).iloc[sorted_idx]

    # outputting the result
    result = {
    "X_test": X_test,
    "y_test": y_test,
    "y_pred": y_pred,
    "resid": resid,
    }
    return result
