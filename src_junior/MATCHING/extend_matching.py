"""Matching groups extension"""
from typing import List, Tuple, Set
from itertools import combinations

def find_matches(element: int, pairs: List[Tuple[int, int]]) -> Set[int]:
    """
    Finding all matches of an element among all the pairs
    """
    matches = [element]
    for pair in pairs:
        # if an element is found on the left of a pair,
        # the right is a match
        if element == pair[0]:
            matches.append(pair[1])
        # vice versa
        elif element == pair[1]:
            matches.append(pair[0])
        else:
            continue

    return set(matches)

def extend_matches(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Predicted matches extension
    (e.g., [(1, 2), (7, 2)] -> [(1, 2), (1, 7), (2, 7)])
    """
    ext_pairs = []
    # going over each pair and matching its elements
    # to the elements of the rest of pairs
    #matches_union = []
    for pair in pairs:
        left_matches = find_matches(pair[0], pairs[:])
        right_matches = find_matches(pair[1], pairs[:])
        # uniting the pair elements' matches
        matches_union = list(left_matches | right_matches)
        # combining into pairs
        matches_pairs = list(combinations(matches_union, 2))
        # saving in one place
        ext_pairs.extend(matches_pairs)

    # saving only unique pairs
    ext_pairs = list(set(ext_pairs))
    # sorting by first and second positions
    ext_pairs = sorted(ext_pairs, key=lambda tup: (tup[0], tup[1]))
    return ext_pairs
