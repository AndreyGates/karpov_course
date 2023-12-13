'''Anti-fraud MLFlow'''
import os
from typing import Any
from typing import Tuple

import fire
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay

IDENTIFIER = f'antifraud-{os.environ.get("KCHECKER_USER_USERNAME", "default")}'
TRACKING_URI = os.environ.get("TRACKING_URI")


def recall_at_precision(
    true_labels: np.ndarray,
    pred_scores: np.ndarray,
    min_precision: float = 0.95,
) -> float:
    """Compute recall at precision

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores
        min_precision (float, optional): Min precision for recall. Defaults to 0.95.

    Returns:
        float: Metric value
    """

    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    metric = max(recall[precision >= min_precision])
    return metric


def recall_at_specificity(
    true_labels: np.ndarray,
    pred_scores: np.ndarray,
    min_specificity: float = 0.95,
) -> float:
    """Compute recall at specificity

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores
        min_specificity (float, optional): Min specificity for recall. Defaults to 0.95.

    Returns:
        float: Metric value
    """

    fpr, tpr, _ = roc_curve(true_labels, pred_scores)
    # tpr = recall, fpr = 1 - specificity (== tnr)
    metric = max(tpr[fpr <= 1 - min_specificity])
    return metric


def curves(true_labels: np.ndarray, pred_scores: np.ndarray) -> Tuple[np.ndarray]:
    """Return ROC and FPR curves

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores

    Returns:
        Tuple[np.ndarray]: ROC and FPR curves
    """

    def fig2numpy(fig: Any) -> np.ndarray:
        fig.canvas.draw()
        img = fig.canvas.buffer_rgba()
        img = np.asarray(img)
        return img

    pr_curve = PrecisionRecallDisplay.from_predictions(true_labels, pred_scores)
    pr_curve = fig2numpy(pr_curve.figure_)

    roc_curve = RocCurveDisplay.from_predictions(true_labels, pred_scores)
    roc_curve = fig2numpy(roc_curve.figure_)

    return pr_curve, roc_curve


def job(
    train_path: str = "",
    test_path: str = "",
    target: str = "target",
):
    """Model training job

    Args:
        train_path (str): Train dataset path
        test_path (str): Test dataset path
        target (str): Target column name
    """
    # setting the tracking url and the experiment name
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(IDENTIFIER)
    # start the experiment
    mlflow.start_run()

    # general tags about model
    task_type = "anti-fraud"
    framework = "sklearn"
    mlflow.set_tags(tags={"task_type": task_type, "framework": framework})

    # loading and logging train and test data
    train_dataset = pd.read_csv(train_path)
    X_train, _ =\
        train_dataset.drop(['target'], axis=1), train_dataset['target']
    test_dataset = pd.read_csv(test_path)
    X_test, y_test =\
        test_dataset.drop(['target'], axis=1), test_dataset['target']

    # log data artifacts (training and test date)
    mlflow.log_artifact(train_path, 'data')
    mlflow.log_artifact(test_path, 'data')

    # log data parameters
    features = X_train.columns.to_list()
    mlflow.log_params(params={"features": features, "target": target})

    # creating, fitting the ML model
    model = IsolationForest(n_estimators=1000)
    #model = OneClassSVM(gamma='auto')
    #model = LocalOutlierFactor(n_neighbors=2)
    model.fit(X_train)

    # log model parameters
    mlflow.log_param("model_type", model.__class__.__name__)

    # predicting
    test_targets = y_test
    pred_scores = -model.score_samples(X_test)

    # calculate and log metrics
    roc_auc = roc_auc_score(test_targets, pred_scores)
    recall_precision_95 = recall_at_precision(test_targets, pred_scores)
    recall_specificity_95 = recall_at_specificity(test_targets, pred_scores)

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("recall_precision_95", recall_precision_95)
    mlflow.log_metric("recall_specificity_95", recall_specificity_95)

    # log pr and roc curves
    pr_curve, roc_curve = curves(test_targets, pred_scores)
    mlflow.log_image(pr_curve, 'metrics/pr.png')
    mlflow.log_image(roc_curve, 'metrics/roc.png')

    # log the model
    mlflow.sklearn.log_model(model,
                             artifact_path=IDENTIFIER,
                             registered_model_name=IDENTIFIER)
    # save the model
    mlflow.sklearn.save_model(model, "model_IsoFor")

    # end the run and logging
    mlflow.end_run()

if __name__ == '__main__':
    from functools import partial

    train_path = 'src_junior/ANTI_FRAUD/train.csv'
    test_path = 'src_junior/ANTI_FRAUD/test.csv'
    fire.Fire(partial(job, train_path, test_path))
