"""Compute constrative loss"""
import numpy as np

def contrastive_loss(
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray, margin: float = 5.0
    )-> float:
    """
    Computes the contrastive loss using numpy.
    Using Euclidean distance as metric function.

    Args:
        x1 (np.ndarray): Embedding vectors of the
            first objects in the pair (shape: (N, M))
        x2 (np.ndarray): Embedding vectors of the
            second objects in the pair (shape: (N, M))
        y (np.ndarray): Ground truthlabels (1 for similar, 0 for dissimilar)
            (shape: (N,))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        float: The contrastive loss
    """
    # calculate the distances for each pair of vectors
    dists = np.linalg.norm(x1-x2, axis=1)
    # calculate the constrative loss for all the objects and average them
    loss = y * dists**2 + np.subtract(1, y) *\
        np.maximum(np.subtract(margin, dists), 0)**2
    return np.mean(loss)
