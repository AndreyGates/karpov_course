"""Solution's template for user."""
from typing import List, Tuple
import numpy as np

def item_neighbors(item: np.ndarray,
                   embeddings: np.ndarray,
                   num_neighbors: int) -> Tuple[np.ndarray, List[float]]:
    """Find all the neighbors of an item.
    
    Parameters
    ----------
    item: np.ndarray :
        the embedding of an input item
    embeddings: np.ndarray :
        embeddings group 
    num_neighbors: int :
        number of neighbors to estimate uniqueness

    Returns
    -------
    Tuple[np.ndarray, List[float]]
        the embeddings of the neighbors and the list of dists

    """
    dists = []
    for index, embed in enumerate(embeddings):
        # distance between the item and every embedding
        dist = np.linalg.norm(embed-item)
        dists.append((index, dist))

    # sorting by dist
    # and getting rid of the first one (item - item = 0)
    dists = sorted(dists, key=lambda tup: tup[1])[1:]
    # extracting the indices top N closest neighbors and the dists
    neighbor_indices = [tup[0] for tup in dists[:num_neighbors]]
    neighbor_dists = [tup[1] for tup in dists[:num_neighbors]]
    # return the neighbors embeds
    neighbors = embeddings[neighbor_indices, :]
    return neighbors, neighbor_dists


def mean_neighbor_distance(dists: List[float]) -> float:
    """Calculate the mean distance from an item to its neighbors.
    
    Parameters
    ----------
    dists: List[float] :
        the corresponding dists between the item and its neighbors

    Returns
    -------
    float:
        the mean distance from an item to its neighbors
     
    """
    mean_distance = np.mean(dists)
    return mean_distance

def knn_uniqueness(embeddings: np.ndarray, num_neighbors: int) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on knn.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group 
    num_neighbors: int :
        number of neighbors to estimate uniqueness    

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    # calculate mean neighbor distance for each item
    mean_dists = []
    for item in embeddings:
        _, neighbor_dists = item_neighbors(item, embeddings, num_neighbors)
        mean_dist = mean_neighbor_distance(neighbor_dists)
        mean_dists.append(mean_dist)

    # return the metric (sku uniqueness = mean neigh distance)
    return np.array(mean_dists)

n = 5
embeddings = np.random.normal(size=(n, 2))
print(embeddings)
print('--------------')
print(knn_uniqueness(embeddings, n))