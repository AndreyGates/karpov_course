"""Statistical tests for residual analysis"""
from typing import Tuple, Optional
import numpy as np

from scipy.stats import shapiro, ttest_1samp, bartlett, levene, fligner

def test_normality(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, bool]:
    """Normality test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the normality test

    is_rejected : bool
        True if the normality hypothesis is rejected, False otherwise

    """
    # resudual terms
    residuals = y_true - y_pred
    # performing Shapiro-Wilk test
    _, p_value = shapiro(residuals)
    is_rejected = bool(p_value < alpha)
    return p_value, is_rejected

def test_unbiased(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefer: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Unbiasedness test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    prefer : str, optional (default=None)
        If None or "two-sided", test whether the residuals are unbiased.
        If "positive", test whether the residuals are unbiased or positive.
        If "negative", test whether the residuals are unbiased or negative.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the test

    is_rejected : bool
        True if the unbiasedness hypothesis is rejected, False otherwise

    """
    # residual terms
    residuals = y_true - y_pred
    # set the T-test parameter
    if prefer is None:
        prefer = 'two-sided'
    elif prefer == 'positive':
        prefer = 'greater'
    elif prefer == 'negative':
        prefer = 'less'

    # performing T-test
    _, p_value = ttest_1samp(residuals, popmean=0.0, alternative=prefer)
    is_rejected = bool(p_value < alpha)
    return p_value, is_rejected

def test_homoscedasticity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: int = 10,
    test: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Homoscedasticity test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    bins : int, optional (default=10)
        Number of bins to use for the test.
        All bins are equal-width and have the same number of samples, except
        the last bin, which will include the remainder of the samples
        if n_samples is not divisible by bins parameter.

    test : str, optional (default=None)
        If None or "bartlett", perform Bartlett's test for equal variances.
        If "levene", perform Levene's test.
        If "fligner", perform Fligner-Killeen's test.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the test

    is_rejected : bool
        True if the homoscedasticity hypothesis is rejected, False otherwise

    """
    # residual terms, sorting them by corresp. ground labels
    # and binning them
    residuals = y_true - y_pred
    sorted_idx = sorted(list(range(len(residuals))), key=lambda i: y_true[i])
    sorted_residuals = residuals[sorted_idx]
    binsets = []
    for i in range(0, len(sorted_residuals), len(sorted_residuals)//bins):
        # if it's the last subset, iterate all the left residuals
        if (len(sorted_residuals) - i) <= bins:
            binset = sorted_residuals[i: ]
        else:
            binset = sorted_residuals[i: i + len(sorted_residuals)//bins]

        binsets.append(binset)

    # performing an ANOVA-test
    if test is None or test == 'bartlett':
        _, p_value = bartlett(*binsets)
    elif test == 'levene':
        _, p_value = levene(*binsets)
    elif test == 'fligner':
        _, p_value = fligner(*binsets)
    else:
        raise ValueError('Wrong value for "test" parameter')

    is_rejected = bool(p_value < alpha)
    return p_value, is_rejected
