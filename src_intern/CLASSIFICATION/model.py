"""Classification model and ROC-AUC"""
from typing import Dict, Tuple

import numpy as np

from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def prepare_data():
    """Generates dataset"""
    X, y = make_classification(
        n_samples=1000, n_features=15, n_informative=10, random_state=42
    )
    return X, y

def solution(data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Function to train a logistic regression model and calculate metrics.

    Parameters:
        data (tuple): Tuple with X and y.

    Returns:
        dict: Dictionary with metrics.

    Examples:
        >>> solution()
        {
            'y_pred': array([0, 1, 1, 0]),
            'y_test': array([0, 1, 1, 0]),
            'roc_auc': 0.99
        }
    """

    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # model fitting
    #model = LogisticRegression(C=0.01)
    model = SVC(C=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)

    return {
        "y_pred": y_pred,
        "y_test": y_test,
        "roc_auc": roc_auc,
    }

if __name__ == "__main__":
    data = prepare_data()
    result = solution(data)
    print(result["roc_auc"])
