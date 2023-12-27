"""Baseline for Kaggle AB."""

from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RepeatedKFold

from scipy import stats

from tqdm import tqdm


def prepare_dataset(DATA_PATH: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare dataset.
    Load data, split into X and y, one-hot encode categorical

    Parameters
    ----------
    DATA_PATH: str :
        path to the dataset

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] :
        X and y
    """
    df = pd.read_csv(DATA_PATH)
    df = df.drop(["ID"], axis=1)
    y = df.pop("y").values

    # select only numeric columns
    X_num = df.select_dtypes(include="number")

    # select only categorical columns and one-hot encode them
    X_cat = df.select_dtypes(exclude="number")
    X_cat = pd.get_dummies(X_cat)

    # combine numeric and categorical
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.fillna(0).values

    return X, y


def single_cv_score(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    cv: Union[int, Tuple[int, int]],
    params: Dict,
    scoring: Callable,
    random_state: int = 42
) -> np.ndarray:
    """
    Cross-validation score for a single model.

    Parameters
    ----------
    model: Callable :
        model to train (e.g. RandomForestRegressor)
    X: np.ndarray :

    y: np.ndarray :

    cv :
        number of folds fo cross-validation

    params: Dict :
        model parameters

    scoring: Callable :
        scoring function (e.g. r2_score)

    random_state: int :
        (Default value = 0)
        random state for cross-validation

    Returns
    -------
    np.ndarray :
        cross-validation scores [n_folds] or [n_repeats x n_folds]

    """
    # altering the model with new parameters
    model.set_params(**params)
    # model scores for each fold
    fold_scores = []
    # setting new params and creating the KFold (or Repeated KFold)
    if isinstance(cv, int):
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    elif isinstance(cv, tuple):
        kf = RepeatedKFold(n_splits=cv[0], n_repeats=cv[1], random_state=random_state)
    else:
        raise ValueError('CV value is not correct!')

    # KFold cross validation predictions for each fold
    # (with direct and inverse convertions)
    for train_index, test_index in kf.split(X):
        # fitting the model on the train set
        # and predicting on the test set
        model.fit(X[train_index], np.log1p(y[train_index]))
        y_pred = model.predict(X[test_index])

        # calculating the score on the test set
        score = scoring(y[test_index], np.expm1(y_pred))
        fold_scores.append(score)

    return np.array(fold_scores)

def cross_val_score(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    cv: Union[int, Tuple[int, int]],
    params_list: List[Dict],
    scoring: Callable,
    random_state: int = 42,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Cross-validation score.

    Parameters
    ----------
    model: Callable :
        model to train (e.g. RandomForestRegressor)
    X: np.ndarray :

    y: np.ndarray :

    cv :
        number of folds fo cross-validation

    params_list: List[Dict] :
        list of model parameters

    scoring: Callable :
        scoring function (e.g. r2_score)

    random_state: int :
        (Default value = 0)
        random state for cross-validation

    show_progress: bool :
        (Default value = False)

    Returns
    -------
    np.ndarray :
        cross-validation scores [n_models x n_folds]

    """
    metrics = []
    rng = tqdm(params_list) if show_progress else params_list
    # going over different models
    for params in rng:
        # KFold cross validation score
        score = single_cv_score(model=model,
                                X=X, y=y,
                                cv=cv, params=params,
                                scoring=scoring,
                                random_state=random_state)
        # save the metric
        metrics.append(score)

    return np.array(metrics)

def compare_models(
    cv: Union[int, Tuple[int, int]],
    model: Callable,
    params_list: List[Dict],
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    alpha: float = 0.05,
    show_progress: bool = False,
) -> List[Dict]:
    """Compare models with Cross val.

    Parameters
    ----------
    cv: Union[int, Tuple[int, int]] :
        (Default value = 5)
        number of folds or (n_folds, n_repeats)
        if int, then KFold is used
        if tuple, then RepeatedKFold is used

    model: Callable :
        model to train (e.g. RandomForestRegressor)

    params_list: List[Dict] :
        list of model parameters

    X: np.ndarray :

    y: np.ndarray :

    random_state: int :
        (Default value = 0)
        random state for cross-validation

    alpha: float :
        (Default value = 0.05)
        significance level for t-test

    show_progress: bool :
        (Default value = False)

    Returns
    -------
    List[Dict] :
        list of dicts with model comparison results
        {
            model_index,
            avg_score,
            p_value,
            effect_sign
        }
    """
    result = []
    # KFold cross-validation for all the models
    metrics = cross_val_score(
        model=model,
        X=X,
        y=y,
        cv=cv,
        params_list=params_list,
        scoring=r2_score,
        random_state=random_state,
        show_progress=show_progress,
    )

    # comparing with the baseline
    baseline_avg_scores = metrics[0]
    for model_index, model_metrics in enumerate(metrics[1:], start=1):
        ttest = stats.ttest_rel(baseline_avg_scores, model_metrics)
        _, p_value = ttest
        # if we can reject Null hypothesis,
        # state the significance of the mean difference
        if p_value < alpha:
            effect_sign = 1\
                if bool(np.mean(model_metrics) > np.mean(baseline_avg_scores))\
                else -1
        # otherwise, there is no statistical effect
        else:
            effect_sign = 0

        result.append({'model_index': model_index,
                       'avg_score': np.mean(model_metrics),
                       'p_value': ttest[1],
                       'effect_sign': effect_sign})

    # sort the model performances by the effect sign
    result = sorted(result, key=lambda d: d['effect_sign'], reverse=True)
    return result

def run() -> None:
    """Run."""

    data_path = "train.csv.zip"
    data_path = "src_junior/KAGGLE_AB/train.csv"
    random_state = 42
    cv = (5, 3)
    params_list = [
        {"max_depth": 10},  # baseline
        {"max_depth": 2},
        {"max_depth": 3},
        {"max_depth": 4},
        {"max_depth": 5},
        {"max_depth": 9},
        {"max_depth": 11},
        {"max_depth": 12},
        {"max_depth": 15},
    ]

    X, y = prepare_dataset(data_path)
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=random_state)

    result = compare_models(
        cv=cv,
        model=model,
        params_list=params_list,
        X=X,
        y=y,
        random_state=random_state,
        show_progress=True,
    )
    print("KFold")
    print(pd.DataFrame(result))
