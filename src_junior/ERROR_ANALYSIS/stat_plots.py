"""Statistical plots"""
import numpy as np
from scipy.stats import norm

def xy_fitted_residuals(y_true, y_pred):
    """Coordinates (x, y) for fitted residuals against true values."""
    # calculating and sorting residuals
    residuals = np.array(y_true) - np.array(y_pred)

    # making the plot
    x_axis = y_pred
    y_axis = np.array(residuals)

    return x_axis, y_axis


def xy_normal_qq(y_true, y_pred):
    """Coordinates (x, y) for normal Q-Q plot."""
    # calculating residuals
    residuals = np.array(y_true) - np.array(y_pred)
    # normally-standardized and sorted residuals
    std_residuals = (residuals-np.mean(residuals)) / np.std(residuals)
    # residuals quantiles and theoretical normal dist. quantiles
    #quantiles = np.percentile(std_residuals, np.linspace(0, 1, 101))
    theoretical_quantiles = norm.ppf(np.linspace(0, 1, len(y_true), endpoint=False))

    # making the plot
    x_axis = np.sort(theoretical_quantiles)
    y_axis = np.sort(std_residuals)

    return x_axis, y_axis


def xy_scale_location(y_true, y_pred):
    """Coordinates (x, y) for scale-location plot."""
    # calculating residuals
    residuals = np.array(y_true) - np.array(y_pred)
    # normally-standardized residuals
    std_residuals = np.abs((residuals-np.mean(residuals))/np.std(residuals))
    root_std_residuals = std_residuals**0.5

    # making the plot
    x_axis = y_pred
    y_axis = root_std_residuals

    return x_axis, y_axis

dist = np.linspace(0, 1, 100, endpoint=False)
percs = norm.ppf(dist)
print(norm.ppf(dist))
print(dist[98] - dist[2])