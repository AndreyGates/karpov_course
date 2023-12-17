"""Constrative loss (using PyTorch)"""
import torch
from torch.nn import PairwiseDistance


def contrastive_loss(
    x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor, margin: float = 5.0
) -> torch.Tensor:
    """
    Computes the contrastive loss using pytorch.
    Using Euclidean distance as metric function.

    Args:
        x1 (torch.Tensor): Embedding vectors of the
            first objects in the pair (shape: (N, M))
        x2 (torch.Tensor): Embedding vectors of the
            second objects in the pair (shape: (N, M))
        y (torch.Tensor): Ground truth labels (1 for similar, 0 for dissimilar)
            (shape: (N,))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        torch.Tensor: The contrastive loss
    """
    # calculate the distances for each pair of vectors
    dists = PairwiseDistance(p=2)(x1, x2)
    # calculate the constrative loss for all the objects and average them
    loss = y * dists**2 + torch.sub(1, y) *\
        torch.maximum(torch.sub(margin, dists), torch.tensor(0))**2
    return torch.mean(loss)
