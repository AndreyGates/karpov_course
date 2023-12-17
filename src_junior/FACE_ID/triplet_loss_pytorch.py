"""Triplet loss (using PyTorch)"""
import torch
from torch.nn import PairwiseDistance

def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 5.0,
) -> torch.Tensor:
    """
    Computes the triplet loss using pytorch.
    Using Euclidean distance as metric function.

    Args:
        anchor (torch.Tensor): Embedding vectors of
            the anchor objects in the triplet (shape: (N, M))
        positive (torch.Tensor): Embedding vectors of
            the positive objects in the triplet (shape: (N, M))
        negative (torch.Tensor): Embedding vectors of
            the negative objects in the triplet (shape: (N, M))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        torch.Tensor: The triplet loss
    """
    # calculate the distances for each pair of vectors
    # (Anchor-Pos and Anchor-Neg)
    ap_dists = PairwiseDistance(p=2)(anchor, positive)
    an_dists = PairwiseDistance(p=2)(anchor, negative)
    # calculate the constrative loss for all the objects and average them
    loss = torch.maximum(torch.tensor(0), ap_dists-an_dists+margin)
    return torch.mean(loss)
