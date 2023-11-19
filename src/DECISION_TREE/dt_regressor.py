'''Modules'''
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import json

@dataclass
class Node:
    """Decision tree node."""
    feature: int = None     # index of a split feature
    threshold: float = None # threshold of a split feature
    n_samples: int = None   # number of rows in a node
    value: float = None     # mean target value of a node
    mse: float = None       # MSE of a node
    left: Node = None       # left node
    right: Node = None      # right node

@dataclass
class DecisionTreeRegressor:
    """Decision tree regressor."""
    max_depth: int
    min_samples_split: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """Build a decision tree regressor from the training set (X, y)."""
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def _mse(self, y: np.ndarray) -> float:
        """Compute the mse criterion for a given set of target values."""
        return np.mean((np.square(np.mean(y) - y)))

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weithed mse criterion for a two given sets of target values"""
        n_left, n_right = y_left.shape[0], y_right.shape[0]
        mse_left, mse_right = self._mse(y_left), self._mse(y_right)
        # MSEs weighted sum
        return (n_left*mse_left + n_right*mse_right) / (n_left + n_right)

    def _split(self, X: np.ndarray, y: np.ndarray, feature: int) -> tuple[float, float]:
        """Find the best split for a node (one feature)"""
        # all the feature's values within
        # the dataset (except for the leftmost and rightmost feature)
        feat_unique_vals = np.unique(X[:, feature])[:-1]
        best_mse, best_threshold = -1.0, None

        for threshold in feat_unique_vals:
            y_left = y[X[:, feature] <= threshold] # left split
            y_right = y[X[:, feature] > threshold] # right split
            current_mse = self._weighted_mse(y_left, y_right) # weighted MSE
            # comparison for finding the best MSE in a feature split
            if best_mse == -1.0 or current_mse < best_mse:
                best_mse, best_threshold = current_mse, threshold

        return best_mse, best_threshold

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for a node."""
        best_mse, best_feature, best_threshold = -1.0, None, None
        # find the best splits for every feature
        for feature in range(X.shape[1]):
            feature_mse, feature_threshold = self._split(X, y, feature)
            #feature_mse = round(feature_mse, 2) # rounding
            # save the best split among all features
            if best_mse == -1.0 or feature_mse < best_mse:
                best_mse, best_feature, best_threshold = feature_mse, feature, feature_threshold

        return best_feature, best_threshold

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split a node and return the resulting left and right child nodes."""
        # creating a node (no matter what is the depth)
        node = Node(
                feature=None,
                threshold=None,
                n_samples=X.shape[0],
                value=round(np.mean(y)),
                mse=self._mse(y),
                left=None,
                right=None
            )
        # if didn't reach the max_depth or min_samples_split, do recursion
        if depth < self.max_depth and node.n_samples >= self.min_samples_split:
            # find the best split for the node (feature and threshold)
            node.feature, node.threshold = self._best_split(X, y)
            # left split and go deep
            left_node_index = X[:, node.feature] <= node.threshold
            X_left, y_left = X[left_node_index], y[left_node_index]
            node.left = self._split_node(X_left, y_left, depth+1)
            # right split and go deep
            right_node_index = X[:, node.feature] > node.threshold
            X_right, y_right = X[right_node_index], y[right_node_index]
            node.right = self._split_node(X_right, y_right, depth+1)

        # reached max_depth - stop recursion
        return node

    def _as_json(self, node: Node) -> str:
        """Return the decision tree as a JSON string. Execute recursively."""
        node_dict = asdict(node)
        node_dict['mse'] = round(node_dict['mse'], 2)
        # base case: if it's a leaf, leave info only about size, mean_y and MSE
        if node.left is None and node.right is None:
            node_dict = {k: v for k, v in node_dict.items()
                         if k in ['value', 'n_samples', 'mse']}
        # otherwise, recurse
        else:
            node_dict = {k: v for k, v in node_dict.items()
                         if k in ['feature', 'threshold', 'n_samples', 'mse']}
            node_dict['left'] = self._as_json(node.left)
            node_dict['right'] = self._as_json(node.right)

        return node_dict

    def as_json(self) -> str:
        """Return the decision tree as a JSON string."""
        js_tree = str(self._as_json(self.tree_)).replace("\'", "\"")
        return js_tree

    def _predict_one_sample(self, features: np.ndarray) -> int:
        """Predict the target value of a single sample."""
        # use dict represantation of a decision tree
        tree_dict = json.loads(self.as_json())
        # extracting the split feature and threshold of a root
        split_feature = tree_dict['feature'] if 'feature' in tree_dict else -1
        split_threshold = tree_dict['threshold'] if 'threshold' in tree_dict else None

        while split_feature != -1: # while not a leaf
            # go down the tree
            if features[split_feature] <= split_threshold:
                tree_dict = tree_dict['left']
            else:
                tree_dict = tree_dict['right']

            # split after going down
            split_feature = tree_dict['feature'] if 'feature' in tree_dict else -1
            split_threshold = tree_dict['threshold'] if 'threshold' in tree_dict else None

        # when a leaf is found, return the leaf mean value
        return tree_dict['value']

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values.
        """
        # applying row-wise predictions
        y = np.apply_along_axis(self._predict_one_sample, 1, X)
        return y
