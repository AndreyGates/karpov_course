'''Modules'''
from typing import List, Tuple

from math import sqrt
from numpy import mean
from scipy.stats import sem, t


def ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Two-sample t-test for the means of two independent samples"""
    # calculate means
    mean1, mean2 = mean(control), mean(experiment)
    # calculate standard errors
    se1, se2 = sem(control), sem(experiment)
    # standard error on the difference between the samples
    sed = sqrt(se1**2.0 + se2**2.0)
    # calculate the t statistic
    t_stat = (mean1 - mean2) / sed
    # degrees of freedom
    df = len(control) + len(experiment) - 2
    # calculate the p-value
    p_value = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    # hypothesis conclusion (True if HA, otherwise HO)
    result = bool(p_value < alpha)
    # return everything
    return p_value, result
