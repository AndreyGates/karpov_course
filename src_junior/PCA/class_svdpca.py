"""PCA class implementation using SVD"""
from typing import Optional
import numpy as np

class SVDPCA:
    """
    Principal Component Analysis (PCA) is a linear dimensionality reduction technique.

    Using Singular Value Decomposition (SVD) to perform PCA.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep.
        If n_components is not set then all components are stored.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

    components_ : ndarray of shape (n_features, n_components)
        Principal axes in feature space, representing the directions of maximum variance.

    explained_variance_ : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If n_components is not set then all components
        are stored and the sum of the ratios is equal to 1.0.
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the model with X by performing eigen decomposition on covariance matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Calculate mean (feature-wise)
        self.mean_ = np.mean(X, axis=0)

        # Centering data
        X_centered = X - self.mean_

        # SVD decomposition
        _, S, Vt = np.linalg.svd(X_centered, full_matrices=True)

        # If no n_components specified, use all the eigenvectors
        n_fit_components = X.shape[1]\
                           if self.n_components is None\
                           else self.n_components
        # Select the first n_components
        self.components_ = Vt.T[:, :n_fit_components]

        # Calculate explained variance
        eigenvalues = S**2
        self.explained_variance_ = eigenvalues[:n_fit_components]
        self.explained_variance_ratio_ =\
            self.explained_variance_ / np.sum(self.explained_variance_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        # Centering data
        X_centered = X - self.mean_
        # Project data
        Y = np.dot(X_centered, self.components_)
        return Y


if __name__ == "__main__":
    X = np.array(
        [
            [1.2, 2, 3.2],
            [4.2, 5.3, 6.6],
            [7, 8.1, 9.7],
            [9, 4, 4.5],
        ]
    )

    print("PCA with all components")
    print("--------")
    pca = SVDPCA()
    pca.fit(X)
    X_projected = pca.transform(X)
    print(pca.__dict__)
    print(pca.explained_variance_ratio_.sum())
    print(X.shape, X_projected.shape)
    print(X_projected)
    print("--------\n")

    print("PCA with 1 component")
    print("--------")
    pca = SVDPCA(1)
    pca.fit(X)
    X_projected = pca.transform(X)
    print(pca.__dict__)
    print(pca.explained_variance_ratio_.sum())
    print(X.shape, X_projected.shape)
    print(X_projected)
    print("--------")
