"""Image posterization"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

@dataclass
class ImageKMeans:
    """Image posterization class"""
    n_clusters: int = 5
    init: str | np.ndarray = "random"
    max_iter: int = 100
    random_state: int = 42

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
                self.centroids_ = X[mask]
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
            elif np.any(self.init < 0) or np.any(self.init > 255):
                msg = "Expected init values to be in [0, 255]"
                raise ValueError(msg)

            # check that init has no duplicate rows
            elif len(np.unique(self.init, axis=0)) != len(self.init):
                msg = "Expected unique rows"
                raise ValueError(msg)

            self.centroids_ = self.init.copy()
        else:
            raise TypeError(f"Unrecognized init type: {self.init}")

    def _assign_centroids(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to the closest centroid"""
        dists = cdist(X, self.centroids_)
        y = np.argmin(dists, axis=1)
        return y

    def fit(self, image: np.ndarray) -> ImageKMeans:
        """Fit k-means to the image"""
        # init K centroids
        X = self._image_as_array(image)
        self._init_centroids(X)

        # iterate until reaching max_iter
        for _ in range(self.max_iter):
            # step 1: centroid assignment
            y = self._assign_centroids(X)
            # step 2: centroid shift
            self._update_centroids(X, y)
            #print(_)

        return self

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return the labels of the image"""
        # convert image to pixel array
        X = self._image_as_array(image)
        # assign each sample to the closest centroid
        labels = self._assign_centroids(X)
        # assign each image pixel to their clusterization label
        image_clusterized = labels.reshape(image.shape[:2])
        return image_clusterized.astype(int)

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Return the compressed image"""
        # predict labels for each pixel
        image_clusterized = self.predict(image)
        # map cluster labels to centroid RGBs
        image_transformed = self.centroids_[image_clusterized]
        return image_transformed.astype(np.uint8)

    def _update_centroids(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the centroids by taking the mean of its samples"""
        labels = np.unique(y)
        for i in range(self.n_clusters):
            # mean over points with the corresp. label
            self.centroids_[i] = X[y == labels[i]].mean(axis=0).astype(int)
