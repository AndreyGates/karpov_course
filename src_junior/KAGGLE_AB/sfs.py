"""Solution for Kaggle AB2."""
from typing import Tuple, List, Any

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

from tqdm import tqdm

class SequentialForwardSelector:
    """
    Sequential forward selection.
    Algorithm selects features one by one, each time adding the feature that
    improves the model the most.

    Parameters
    ----------
    model: estimator :
        ML model, e.g. LinearRegression
    cv: cross-validation generator :
        cross-validation generator, e.g. KFold, RepeatedKFold
    max_features: int :
        maximum number of features to select
    verbose: int :
        (Default value = 0)
        verbosity level
    

    Attributes
    ----------
    n_features_: int :
        number of features in the dataset
    selected_features_: List[int] :
        list of selected features, ordered by index
    n_selected_features_: int :
        number of selected features
    """

    def __init__(
        self,
        model,
        cv,
        max_features: int = 10,
        verbose: int = 0,
    ) -> None:
        """Initialize SequentialForwardSelector."""
        # number of input features
        self.n_features_: int = None
        # selected features (after SFS)
        self.selected_features_: List[int] = []
        # the model
        self.model: Any = model
        # the cross validator
        self.cv: Any = cv
        # the max number of features to be selected
        self.max_features: int = max_features
        # the verbose state
        self.verbose: int = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model.

        Parameters
        ----------
        X: np.ndarray :
            features
        y: np.ndarray :
            target
        """
        self.n_features_ = X.shape[1]
        # included and excluded featuresets
        included, excluded = set(), set(range(self.n_features_))

        # doing SFS: going over each excluded feature
        # verbosity (for the outer loop)
        rng_out = range(self.max_features) if self.verbose == 0\
                else tqdm(range(self.max_features))
        # score savers
        global_score = -np.inf
        for _ in rng_out:
            # verbosity (for the inner loop)
            rng_in = excluded if self.verbose == 0 else tqdm(excluded)

            # optimization step (inner loop)
            # reset "max_score" before the next optimization step
            max_score, argmax_feature = -np.inf, -1
            for feature in rng_in:
                # calculate the score on selected data
                new_included = list(included) + [feature]
                X_reduced = X[:, new_included]
                scores = cross_val_score(self.model,
                                            X_reduced, y,
                                            scoring="r2",
                                            cv=self.cv, n_jobs=-1)

                # if the new feature increased the cum. model perf, save it
                if scores.mean() > max_score:
                    argmax_feature = feature
                    max_score = scores.mean()

            # if the optimization step gave out a feature
            # giving a better general perf
            if argmax_feature != -1 and max_score > global_score:
                # adding the best feature to the included set
                included.add(argmax_feature)
                # excluding it from the corresp. set
                excluded.remove(argmax_feature)
                # update the global score
                global_score = max_score

            # if no improvement after adding the best feature, terminate
            else:
                break

        # saving the selected features
        self.selected_features_ = sorted(list(included))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce the dataset to selected features.

        Parameters
        ----------
        X: np.ndarray :
            features

        Returns
        -------
        X: np.ndarray :
            reduced dataset
        """
        X_reduced = X[:, self.selected_features_]
        return X_reduced

    @property
    def n_selected_features_(self):
        """Number of selected features."""
        if self.selected_features_ is not None:
            return len(self.selected_features_)
        raise ValueError('There are no selected features!')

def generate_dataset(
    n_samples: int = 10_000,
    n_features: int = 50,
    n_informative: int = 10,
    random_state: int = 42,
) -> Tuple:
    """
    Generate dataset.

    Parameters
    ----------
    n_samples: int :
        (Default value = 10_000)
        number of samples
    n_features: int :
        (Default value = 50)
        number of features
    n_informative: int :
        (Default value = 10)
        number of informative features, other features are noise
    random_state: int :
        (Default value = 42)
        random state for reproducibility

    Returns
    -------
    X: np.ndarray :
        features
    y: np.ndarray :
        target

    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=100,
        random_state=random_state,
        n_informative=n_informative,
        bias=100,
        shuffle=True,
    )
    return X, y


def run() -> None:
    """Run."""
    random_state = 79
    n_samples = 10_000
    n_features = 50
    n_informative = 5
    max_features = 10
    n_splits = 3
    n_repeats = 10

    # generate data
    X, y = generate_dataset(n_samples, n_features, n_informative, random_state)

    # define model and cross-validation
    model = LinearRegression()
    cv = RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    # baseline
    scores = cross_val_score(model, X, y, scoring="r2", cv=cv, n_jobs=-1)
    print(f"Baseline features count: {X.shape[1]}")
    print(f"Baseline R2 score: {scores.mean():.4f}")

    selector = SequentialForwardSelector(model, cv, max_features, verbose=1)
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    scores = cross_val_score(model, X_transformed, y, scoring="r2", cv=cv, n_jobs=-1)

    print(f"Features: {selector.selected_features_}")
    print(f"Features count: {selector.n_selected_features_}")
    print(f"Mean R2 score: {scores.mean():.4f}")


if __name__ == "__main__":
    run()
