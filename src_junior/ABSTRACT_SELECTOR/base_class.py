"""Abstract selector"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

class BaseSelector(ABC):
    '''Base selector class'''
    @abstractmethod
    def fit(self):
        """Base fitting method"""
        return

    def transform(self, X):
        """Transforms input data"""
        return X[self.selected_features]

    def fit_transform(self, X, y):
        """Fits and transforms input date"""
        self.fit(X, y)
        return self.transform(X)

    @property
    def n_features_(self):
        """Number of selected features"""
        return len(self.selected_features)

    @property
    def original_features_(self):
        """Input features"""
        return self.original_features

    @property
    def selected_features_(self):
        """Selected features"""
        return self.selected_features

@dataclass
class PearsonSelector(BaseSelector):
    '''Pearson selector'''
    threshold: float = 0.5

    def fit(self, X, y) -> PearsonSelector:
        """Fitting method"""
        # Correlation between features and target
        corr = pd.concat([X, y], axis=1).corr(method="pearson")
        corr_target = corr.iloc[:-1, -1]

        self.original_features = X.columns.tolist()
        self.selected_features = corr_target[
            abs(corr_target) >= self.threshold
        ].index.tolist()

        return self

@dataclass
class SpearmanSelector(BaseSelector):
    '''Spearman selector'''
    threshold: float = 0.5

    def fit(self, X, y) -> SpearmanSelector:
        """Fitting method"""
        corr = pd.concat([X, y], axis=1).corr(method="spearman")
        corr_target = corr.iloc[:-1, -1]
        self.original_features = X.columns.tolist()
        self.selected_features = corr_target[
            abs(corr_target) >= self.threshold
        ].index.tolist()

        return self

@dataclass
class VarianceSelector(BaseSelector):
    '''Variance selector'''
    min_var: float = 0.4

    def fit(self, X, y=None) -> VarianceSelector:
        """Fitting method"""
        variances = np.var(X, axis=0)
        self.original_features = X.columns.tolist()
        self.selected_features = X.columns[variances > self.min_var].tolist()
        return self

X = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
X_selected = VarianceSelector().fit_transform(X, y=None)
print(X_selected)