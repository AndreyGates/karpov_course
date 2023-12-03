"""Solution for Similar Items task"""
from typing import Dict
from typing import List
from typing import Tuple

from itertools import combinations
import numpy as np
from scipy.spatial.distance import cosine


class SimilarItems:
    """Similar items class"""

    @staticmethod
    def similarity(embeddings: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], float]:
        """Calculate pairwise similarities between each item
        in embedding.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.

        Returns:
            Tuple[List[str], Dict[Tuple[int, int], float]]:
            List of all items + Pairwise similarities dict
            Keys are in form of (i, j) - combinations pairs of item_ids
            with i < j.
            Round each value to 8 decimal places.
        """
        pair_sims = {}
        # iterate over each pair of items
        for pair in list(combinations(embeddings.keys(), 2)):
            # for each pair calculate the cosine sim
            similarity = 1 - cosine(embeddings[pair[0]], embeddings[pair[1]])
            pair_sims[pair] = round(similarity, 8)

        return pair_sims

    @staticmethod
    def knn(
        sim: Dict[Tuple[int, int], float], top: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Return closest neighbors for each item.

        Args:
            sim (Dict[Tuple[int, int], float]): <similarity> method output.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, List[Tuple[int, float]]]: Dict with top closest neighbors
            for each item.
        """
        keys, values = list(sim.keys()), list(sim.values())
        # sorting indices by values (ASC)
        sorted_value_index = np.argsort(values)
        # creating a sorted dict of all neighbor pairs (TOP DESC)
        sorted_sim_dict = {keys[i]: values[i] for i in sorted_value_index[::-1]}
        # creating top neighbor lists for each item
        knn_dict = {}
        for pair, similarity in sorted_sim_dict.items():
            # making a list of top neighbors for the left and right elements
            for k in [0, 1]:
                # if the item is not accounted yet, add it
                if pair[k] not in knn_dict:
                    knn_dict[pair[k]] = []
                # add new neighbors as far as top N
                if len(knn_dict[pair[k]]) < top:
                    knn_dict[pair[k]].append((pair[1-k], similarity))

        # sort the dict by item indides for further convenience
        indices = list(knn_dict.keys())
        indices.sort()
        knn_dict = {i: knn_dict[i] for i in indices}

        return knn_dict

    @staticmethod
    def knn_price(
        knn_dict: Dict[int, List[Tuple[int, float]]],
        prices: Dict[int, float],
    ) -> Dict[int, float]:
        """Calculate weighted average prices for each item.
        Weights should be positive numbers in [0, 2] interval.

        Args:
            knn_dict (Dict[int, List[Tuple[int, float]]]): <knn> method output.
            prices (Dict[int, float]): Price dict for each item.

        Returns:
            Dict[int, float]: New prices dict, rounded to 2 decimal places.
        """
        knn_price_dict = {}
        # iterate over items and their top neighbors with sims
        for item, neighbor_sims in knn_dict.items():
            # calculating the sum of [cosine + 1] with each neighbor of an item
            weight_sum = sum((neigh_sim[1]+1) for neigh_sim in neighbor_sims)
            # calculating the weighed price for an item (weight are 1-normalized)
            knn_price_dict[item] =\
                                round(
                                sum(((neigh_sim[1]+1) / weight_sum)\
                                *prices[neigh_sim[0]] for neigh_sim in neighbor_sims),
                                2)
        return knn_price_dict

    @staticmethod
    def transform(
        embeddings: Dict[int, np.ndarray],
        prices: Dict[int, float],
        top: int,
    ) -> Dict[int, float]:
        """Transforming input embeddings into a dictionary
        with weighted average prices for each item.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.
            prices (Dict[int, float]): Price dict for each item.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, float]: Dict with weighted average prices for each item.
        """
        # 1: pairwise item similarities
        pair_sims = SimilarItems.similarity(embeddings=embeddings)
        # 2: top N neighbors for each item
        knn_dict = SimilarItems.knn(sim=pair_sims, top=top)
        # 3: new prices for each item
        knn_price_dict = SimilarItems.knn_price(knn_dict=knn_dict, prices=prices)

        return knn_price_dict
