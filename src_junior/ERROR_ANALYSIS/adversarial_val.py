"""Adversarial validation"""
from typing import Optional

import pandas as pd
import numpy as np
import residuals
from sklearn.base import ClassifierMixin

from sklearn.metrics import roc_auc_score

def adversarial_validation(
    classifier: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    quantile: float = 0.1,
    func: Optional[str] = None,
) -> dict:
    """Adversarial validation residual analysis"""
    # residuals calculation (saving absolute values)
    if func is None: # if no specific func, apply subtraction
        func = 'residuals'
    resid_func = getattr(residuals, func)
    resid = pd.Series(np.abs(resid_func(y_test, y_pred)))

    # sorting data starting with the best cases (= smallest residuals)
    sorted_idx = np.argsort(resid)
    sorted_resid = resid.iloc[sorted_idx]

    # threshold for which larger residuals are considered "bad" class
    #sorted_resid = resid[sorted_idx]
    #threshold = np.quantile(sorted_resid, quantile)
    #top_k = len(sorted_resid[sorted_resid > threshold])

    # "bad" and "good" datasets (with quantile cut)
    #X_adversarial = X_test.iloc[sorted_idx]
    X_adversarial = X_test.copy()

    is_error = resid > sorted_resid.quantile(1-quantile)
    y_adversarial = is_error.astype(int)
    #top_k = np.floor(len(X_test) * quantile).astype(int)
    #y_adversarial = np.array([1]*top_k + [0]*(len(resid)-top_k))

    # adversarial validator
    classifier.fit(X_adversarial, y_adversarial)
    # positive class probabilities
    adversarial_preds = np.array(classifier.predict_proba(X_adversarial))[:, 1]

    # ROC AUC
    roc_auc = roc_auc_score(y_adversarial, adversarial_preds)

    # feature importances
    if hasattr(classifier, 'feature_importances_'):
        feature_importances = pd.Series(classifier.feature_importances_,
                                           index=X_adversarial.columns)
        #feature_importances = np.abs(feature_importances)
    elif hasattr(classifier, 'coef_'):
        feature_importances = pd.Series(np.abs(classifier.coef_)[0, :],
                                           index=X_adversarial.columns)
    else:
        feature_importances = None

    result = {
        "ROC-AUC": roc_auc,
        "feature_importances": feature_importances,
    }

    return result
