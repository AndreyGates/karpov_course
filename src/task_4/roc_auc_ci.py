'''Modules'''
from typing import Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score

def roc_auc_ci(
    classifier: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    conf: float = 0.95,
    n_bootstraps: int = 10_000,
) -> Tuple[float, float]:
    """Returns confidence bounds of the ROC-AUC"""
   
    y_pred = classifier.predict_proba(X)[:, 1] # classifier predictions
    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    # sorting ROC AUC scores for creating CI
    sorted_scores.sort()

    # Computing the lower and upper bound of the input confidence
    confidence_lower = sorted_scores[int((1-conf) / 2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int((1+conf) / 2 * len(sorted_scores))]

    return (confidence_lower, confidence_upper)
