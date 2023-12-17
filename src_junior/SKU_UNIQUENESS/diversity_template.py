"""Template for user."""
from typing import Tuple
import numpy as np
from sklearn.neighbors import KernelDensity


def kde_uniqueness(embeddings: np.ndarray) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on KDE.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group 

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    # fitting the density acculumlation model and finging negative log densities
    kde = KernelDensity(kernel='gaussian').fit(embeddings)
    log_density = kde.score_samples(embeddings)
    # calculate the uniqueness metric
    # (must be positive and inversely proportional to the density)
    uniqueness = 1 / np.exp(log_density)
    return uniqueness

def group_diversity(embeddings: np.ndarray, threshold: float) -> Tuple[bool, float]:
    """Calculate group diversity based on kde uniqueness.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group
    threshold: float :
       group deversity threshold for reject group

    Returns
    -------
    Tuple[bool, float]
        reject
        group diverstity

    """
    # calculate KDE uniqueness for each item
    uniqueness = kde_uniqueness(embeddings)
    # sum it up into the group diversity
    group_div = uniqueness.sum()
    # report the results (return True if the threshold is not passed)
    return group_div < threshold, group_div
