"""Solution's template for user."""
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
