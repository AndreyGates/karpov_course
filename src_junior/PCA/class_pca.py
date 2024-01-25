"""PCA class implementation"""
from typing import Optional
import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) is a linear dimensionality reduction technique.

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
        If n_components is not set then all components are stored
        and the sum of the ratios is equal to 1.0.
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components

    def fit(self, X: np.ndarray):
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

        # Calculate covariance matrix
        n = X.shape[0]
        cov_matrix = np.dot(X_centered.T, X_centered) / (n-1)
        #cov_matrix = np.cov(X_centered, rowvar=False)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvectors based on eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        sorted_vectors = eigenvectors[:, idx]
        sorted_eigenvalues = eigenvalues[idx]

        # If no n_components specified, use all the eigenvectors
        n_fit_components = X.shape[1]\
                           if self.n_components is None\
                           else self.n_components

        # Eigenvectors sorted by variance degree
        self.components_ = sorted_vectors[:, :n_fit_components]

        # Calculate explained variance
        self.explained_variance_ = sorted_eigenvalues[:n_fit_components]
        self.explained_variance_ratio_ =\
            sorted_eigenvalues[:n_fit_components] / np.sum(sorted_eigenvalues)

    def transform(self, X):
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
    #X = np.array([[0.5, 0.8, 0.8, 0.1, 0.5, 0.9], [0.5, 0.4, 0.6, 0.9, 0., 0.5], [0.2, 0.1, 0.8, 0.9, 0.8, 0.], [0., 0.2, 0.9, 0.1, 0.2, 0.5]])

    print("PCA with all components")
    print("--------")
    pca = PCA()
    pca.fit(X)
    X_projected = pca.transform(X)
    print(pca.__dict__)
    print(pca.explained_variance_ratio_.sum())
    print(X.shape, X_projected.shape)
    print(X_projected)
    print("--------\n")

    print("PCA with 1 component")
    print("--------")
    pca = PCA(1)
    pca.fit(X)
    X_projected = pca.transform(X)
    print(pca.__dict__)
    print(pca.explained_variance_ratio_.sum())
    print(X.shape, X_projected.shape)
    print(X_projected)
    print("--------")
