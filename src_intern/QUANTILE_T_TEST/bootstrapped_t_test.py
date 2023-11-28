'''Modules'''
from typing import List, Tuple

import numpy as np
from scipy.stats import ttest_ind


def bootstrapped_percentile(
        x: List[float],
        quantile: float = 0.95,
        n_bootstraps: int = 1000
        ) -> List[float]:
    """
    Bootstrapped percentile distribution
    """
    bootstrapped_quantiles = []

    for _ in range(n_bootstraps):
        # bootstrap the percentile of a dataset
        bootstrapped_sample = np.random.choice(x, size=len(x), replace=True)
        bootstrapped_quantiles.append(np.percentile(bootstrapped_sample, quantile*100))

    return bootstrapped_quantiles

def quantile_ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
    quantile: float = 0.95,
    n_bootstraps: int = 1000,
) -> Tuple[float, bool]:
    """
    Bootstrapped t-test for quantiles of two samples.
    """
    # bootstrapping the quantile for CNTRL and EXP sets
    control_quantiles = bootstrapped_percentile(control, quantile, n_bootstraps)
    experiment_quantiles = bootstrapped_percentile(experiment, quantile, n_bootstraps)

    # t-statistic for the quantile datasets
    t_test = ttest_ind(control_quantiles, experiment_quantiles)
    # p_value and HA/H0
    _, p_value = t_test
    result = bool(p_value < alpha)

    return p_value, result
