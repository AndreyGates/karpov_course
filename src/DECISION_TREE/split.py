'''Modules'''
from __future__ import annotations
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

def split(X: np.ndarray, y: np.ndarray, feature: int) -> tuple[float, float]:
    """Find the best split for a node (one feature)"""
    # all the feature's values within
    # the dataset (except for the leftmost and rightmost)
    feat_unique_vals = np.unique(X[:, feature])[1:-1]
    best_mse, best_threshold = -1.0, None

    for threshold in feat_unique_vals:
        y_left = y[X[:, feature] <= threshold] # left split
        y_right = y[X[:, feature] > threshold] # right split
        current_mse = weighted_mse(y_left, y_right) # weighted MSE
        # comparison for finding the best MSE in a feature split
        if best_mse == -1.0 or current_mse < best_mse:
            best_mse, best_threshold = current_mse, threshold

    return best_mse, best_threshold

def best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Find the best split for a node (one feature)"""
    best_mse, best_feature, best_threshold = -1.0, None, None
    # find the best splits for every feature
    for feature in range(X.shape[1]):
        feature_mse, feature_threshold = split(X, y, feature)
        # save the best split among all features
        if best_mse == -1.0 or feature_mse < best_mse:
            best_mse, best_feature, best_threshold = feature_mse, feature, feature_threshold

    return best_feature, best_threshold
