"""Triplet loss"""
import numpy as np

def triplet_loss(
    anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float = 5.0
) -> float:
    """
    Computes the triplet loss using numpy.
    Using Euclidean distance as metric function.

    Args:
        anchor (np.ndarray): Embedding vectors of
            the anchor objects in the triplet (shape: (N, M))
        positive (np.ndarray): Embedding vectors of
            the positive objects in the triplet (shape: (N, M))
        negative (np.ndarray): Embedding vectors of
            the negative objects in the triplet (shape: (N, M))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        float: The triplet loss
    """
    # calculate the distances for each pair of vectors
    # (Anchor-Pos and Anchor-Neg)
    ap_dists = np.linalg.norm(anchor-positive, axis=1)
    an_dists = np.linalg.norm(anchor-negative, axis=1)
    # calculate the constrative loss for all the objects and average them
    loss = np.maximum(0, ap_dists-an_dists+margin)
    return np.mean(loss)
