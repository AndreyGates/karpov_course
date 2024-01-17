"""Image posterization"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import distance


@dataclass
class ImageKMedoids:
    """Image posterization class"""
    n_clusters: int = 5
    init: str | np.ndarray = "random"
    max_iter: int = 100
    random_state: int = 42

    def fit(self, image: np.ndarray) -> ImageKMedoids:
        """Fit the model to the image"""
        # convert image to list of RGB values
        X = image.reshape(-1, 3)
        # select N random samples as initial centroids
        self._init_centroids(X)

        # iterate until convergence

        for _ in range(self.max_iter):
            # assign each sample to the closest medois
            y = self._assign_centroids(X)
            # update medoids
            self._update_centroids(X, y)
            #print(_)

        return self

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return the labels of the image"""
        # convert image to list of RGB values
        X = self._image_as_array(image)
        # assign each sample to the closest centroid
        labels = self._assign_centroids(X)
        # convert labels to matrix
        return labels.reshape(image.shape[:2])

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Return the compressed image"""
        labels = self.predict(image)
        image_compressed = self.centroids_[labels.astype(int)]
        return image_compressed.astype(np.uint8)

    def _image_as_array(self, image: np.ndarray) -> np.ndarray:
        """Convert image to list of RGB values"""
        return image.reshape(-1, 3)

    def _init_centroids(self, X: np.ndarray) -> None:
        """Select N random samples as initial centroids"""
        if isinstance(self.init, str):
            if self.init == "random":
                # set random seed
                np.random.seed(self.random_state)

                # select N random samples as initial centroids
                mask = np.random.choice(len(X), self.n_clusters, replace=False)
                self.centroids_ = X[mask].copy()
            else:
                raise ValueError(f"Unrecognized str init: {self.init}")
        elif isinstance(self.init, np.ndarray):
            # check that init has the correct shape
            shape_expected = (self.n_clusters, 3)
            if self.init.shape != shape_expected:
                shape = self.init.shape
                msg = f"Expected init shape {shape_expected}, got: {shape}"
                raise ValueError(msg)

            # check that init has the correct values
            if np.any(self.init.flatten() < 0):
                msg = "Expected init values to be in [0, 255]"
                raise ValueError(msg)

            if np.any(self.init.flatten() > 255):
                msg = "Expected init values to be in [0, 255]"
                raise ValueError(msg)

            # check that init has unique values
            if len(np.unique(self.init, axis=0)) != self.n_clusters:
                msg = "Expected init to have unique values"
                raise ValueError(msg)

            self.centroids_ = self.init.copy()
        else:
            raise TypeError(f"Unrecognized init type: {self.init}")

    def _update_centroids(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the centroids by min L1-loss"""
        labels = np.unique(y)
        for i in range(self.n_clusters):
            cluster_samples = X[y == labels[i]]

            # pairwise distances for cluster elements (with memory regulation)
            sub_cluster_size = 1000
            # we only choose several random points as the potential memoids
            if len(cluster_samples) > sub_cluster_size:
                mask = np.random.choice(len(cluster_samples), sub_cluster_size, replace=False)
                sub_cluster = cluster_samples[mask]
                cluster_losses = distance.cdist(cluster_samples,
                                                sub_cluster,
                                                'cityblock').sum(axis=1)
            else:
                cluster_losses = distance.cdist(cluster_samples,
                                                cluster_samples,
                                                'cityblock').sum(axis=1)
            # update centroid
            self.centroids_[i] = cluster_samples[np.argmin(cluster_losses)]

    def _assign_centroids(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to the closest centroid"""
        # compute the distance between each sample and each centroid
        dist = distance.cdist(X, self.centroids_, metric="cityblock")
        # assign each sample to the closest centroid
        y = np.argmin(dist, axis=1)
        return y
