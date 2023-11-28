'''Modules'''
from functools import partial
from typing import Tuple, Callable

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, precision_score, recall_score

def score_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    score_func: Callable,
    threshold: float,
    conf: float = 0.95,
    n_bootstrap: int = 10_000,
) -> Tuple[float, float]:
    """Returns confidence bounds of a statistic score"""
    bootstrapped_scores = []

    for _ in range(n_bootstrap):
        # bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = score_func(y_true[indices], y_prob[indices] > threshold)
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    # sorting ROC AUC scores for creating CI
    sorted_scores.sort()

    # Computing the lower and upper bound of the input confidence
    confidence_lower = sorted_scores[int((1-conf) / 2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int((1+conf) / 2 * len(sorted_scores))]

    return (confidence_lower, confidence_upper)

def pr_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float,
) -> Tuple[float, float]:
    """Returns threshold and recall (from Precision-Recall Curve)"""

    # analyze precision-recall curve (starting with the highest recall and lowest precision)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    max_recall = -0.1
    threshold_proba = 1.0
    for prec, rec, thresh in zip(precision, recall, thresholds):
        # if we reached the min prec, return the rec and threshold
        if prec >= min_precision and max_recall < rec:
            threshold_proba, max_recall = thresh, rec
            break

    return threshold_proba, max_recall


def sr_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_specificity: float,
) -> Tuple[float, float]:
    """Returns threshold and recall (from Specificity-Recall Curve)"""

    # analyze specificity-recall curve (with FPR = 1 - specificity and TPR = recall in asc. order)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    max_recall = -0.1
    threshold_proba = 1.0
    for spec, rec, thresh in list(zip(1-fpr, tpr, thresholds))[::-1]:
        # if we reached the min spec, return the rec and threshold
        if spec >= min_specificity and max_recall < rec:
            threshold_proba, max_recall = thresh, rec
            break

    return threshold_proba, max_recall


# random generator for bootstrapping
rng = np.random.default_rng()

def pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns Precision-Recall curve and it's (LCB, UCB)"""
    # obtain precision-recall curve (starting with the highest recall and lowest precision)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # LCBs and UCBs for precision
    precision_lcb, precision_ucb = [], []
    for threshold in thresholds:
        lcb, ucb = score_ci(y_true, y_prob,
                            precision_score,
                            threshold, conf=conf, n_bootstrap=n_bootstrap)
        precision_lcb.append(lcb)
        precision_ucb.append(ucb)

    return recall, precision, np.array(precision_lcb), np.array(precision_ucb)


def sr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns Specificity-Recall curve and it's (LCB, UCB)"""
    # obtain specificity-recall curve (with FPR = 1 - specificity and TPR = recall in asc. order)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity, recall = 1-fpr, tpr

    # LCBs and UCBs for specificity
    specificity_lcb, specificity_ucb = [], []
    for threshold in thresholds:
        lcb, ucb = score_ci(y_true, y_prob,
                            partial(recall_score, pos_label=0),
                            threshold, conf=conf, n_bootstrap=n_bootstrap)
        specificity_lcb.append(lcb)
        specificity_ucb.append(ucb)

    return recall, specificity, np.array(specificity_lcb), np.array(specificity_ucb)
