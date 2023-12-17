'''Recommendation REST-API service'''
from typing import Tuple

import os
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every

from sklearn.neighbors import KernelDensity


DIVERSITY_THRESHOLD = 10

app = FastAPI()
embeddings = {}


@app.on_event("startup")
@repeat_every(seconds=10)
def load_embeddings() -> dict:
    """Load embeddings from file."""

    # Load new embeddings each 10 seconds
    np.save(os.path.dirname(__file__)+'/embeddings.npy',
            np.array({0: [0, 0],
                      1: [1, 1],
                      2: [2, 2],
                      3: [3, 3],
                      4: [0.6642812619278889, -0.9179558157827237],
                      5: [-0.5623831071085313, -0.7229905723309549],
                      6: [-0.6508344464097425, 1.2610313683916736],
                      7: [0.25513670775500347, 0.8586544165575627],
                      8: [-0.8941995718037754, -0.3408241513566439],
                      9: [0.9191719955127744, 0.5886520432771619]}))
    path = os.path.join(os.path.dirname(__file__), "embeddings.npy")
    embeddings_raw = np.load(path, allow_pickle=True).item()
    for item_id, embedding in embeddings_raw.items():
        embeddings[item_id] = embedding

    return {}


@app.get("/uniqueness/")
def uniqueness(item_ids: str) -> dict:
    """Calculate uniqueness of each product"""
    # Parse item IDs
    item_ids = [int(item) for item in item_ids.split(",")]

    # Default answer
    item_uniqueness = {item_id: 0.0 for item_id in item_ids}

    # Calculate uniqueness
    item_embeddings = []
    for item_id in item_ids:
        # extract all the input items embeds
        item_embeddings.append(embeddings[item_id])

    uniqness = kde_uniqueness(item_embeddings)
    # updating uniqueness for each input item
    item_uniqueness = dict(zip(item_ids, uniqness))

    return item_uniqueness


@app.get("/diversity/")
def diversity(item_ids: str) -> dict:
    """Calculate diversity of group of products"""
    # Parse item IDs
    item_ids = [int(item) for item in item_ids.split(",")]

    # Default answer
    answer = {"diversity": 0.0, "reject": True}

    # Calculate diversity
    item_embeddings = []
    for item_id in item_ids:
        # extract all the input items embeds
        item_embeddings.append(embeddings[item_id])

    reject, div = group_diversity(item_embeddings, threshold=DIVERSITY_THRESHOLD)
    # return the service response
    answer["diversity"], answer["reject"] = float(div), bool(reject)
    return answer


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
    # Fit a kernel density estimator to the item embedding space
    kde = KernelDensity().fit(embeddings)

    uniqness = []
    for item in embeddings:
        uniqness.append(1 / np.exp(kde.score_samples([item])[0]))

    return np.array(uniqness)


def group_diversity(embeddings: np.ndarray,
                    threshold: float) -> Tuple[bool, float]:
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
    div = np.sum(kde_uniqueness(embeddings)) / len(embeddings)
    reject = div < threshold
    return reject, div


def main() -> None:
    """Run application"""
    uvicorn.run("main:app", host="localhost", port=5000)


if __name__ == "__main__":
    main()
