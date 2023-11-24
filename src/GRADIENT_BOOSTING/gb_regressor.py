'''Modules'''
from typing import List, Tuple, Callable

import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    """Gradient boosting regressor."""
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        subsample_size=0.1,
        replace=False,
        loss="mse",
        verbose=False
    ):
        # number of a model's trees
        self.n_estimators_ = n_estimators
        # learning rate
        self.learning_rate = learning_rate
        # maximum depth of a tree
        self.max_depth = max_depth
        # minimum number of leaves needed to split a tree
        self.min_samples_split = min_samples_split
        # to include logging or not (TRUE or FALSE)
        self.verbose = verbose
        # subsampling for fit()
        self.subsample_size = subsample_size
        # with or without replacement
        self.replace = replace

        # list of learning trees
        self.trees_: List[DecisionTreeRegressor] = []
        # base model prediction (average value)
        self.base_pred_: float = ''

        # defining the loss function
        # if a str was passed
        if loss in ['mse']:
            self.loss = self._mse
        # if a function was passed
        elif isinstance(loss, Callable):
            self.loss = loss
        else:
            raise ValueError('The input function is not recognized')

    def _mse(self, y_true, y_pred) -> Tuple[float, np.ndarray]:
        """Mean squared error loss function and gradient."""
        loss = np.mean(np.square(y_pred - y_true)) # quadratic loss
        grad = y_pred - y_true # gradient (vector taking all parameters)
        return loss, grad

    def _subsample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: 
        '''
        Returns the subset of data 
        (with/without replacement)
        '''
        size = int(len(X)*self.subsample_size)
        if size == 0: # if subsample_len equals 0, add 1
            size += 1

        indices = np.random.choice(len(X),
                                   size=size,
                                   replace=self.replace)
        sub_X, sub_y = X[indices, :], y[indices]
        return sub_X, sub_y

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
        """
        # base (initial mean) and full predictions
        self.base_pred_ = np.mean(y, axis=0)
        full_prediction = np.full(len(X), self.base_pred_)
        # creating a chain of trees and learn subsequently
        for _ in range(self.n_estimators_):
            # calculating preudo residuals as the gradient
            loss, pseudo_residuals = self.loss(y, full_prediction)
            # if needed, print the iteration's MSE
            if self.verbose:
                print(f'Training MSE = {loss}')
            # creating and fitting a tree
            dt_regressor = DecisionTreeRegressor(max_depth=self.max_depth,
                                                 min_samples_split=self.min_samples_split)

            # subsampling X and antigrad for stochastic learning
            X_sub, pseudo_residuals_sub = self._subsample(X.copy(), pseudo_residuals.copy())
            # fitting the antigradients
            dt_regressor.fit(X_sub, -pseudo_residuals_sub)
            # gradient boost predictions (scaling by learning rate)
            tree_prediction = self.learning_rate * dt_regressor.predict(X)
            # adding them to the full prediction
            full_prediction += tree_prediction
            # saving the tree
            self.trees_.append(dt_regressor)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            y: array-like of shape (n_samples,)
            The predict values.
            
        """
        # the base prediction
        full_prediction = np.full(len(X), self.base_pred_)
        # going over all the learning trees to compose the full pred
        for tree in self.trees_:
            tree_prediction = tree.predict(X)
            # adding scaled gradient boost pred
            full_prediction += self.learning_rate * tree_prediction

        return full_prediction
