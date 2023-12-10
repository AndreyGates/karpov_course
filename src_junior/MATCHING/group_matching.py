"""Group matching"""
from typing import List
from itertools import combinations

def is_linkable(groups: List[tuple]) -> bool:
    """Checking if any two groups are still linkable or not"""
    comb_groups = list(combinations(groups, 2))
    for comb in comb_groups:
        # in case of intersection
        if set(comb[0]) & set(comb[1]) != set():
            return True

    return False

def extend_matches(groups: List[tuple]) -> List[tuple]:
    """Extending match clusters into united groups"""
    ext_groups = groups[:]
    # sorting already existing groups
    for i, group in enumerate(ext_groups):
        ext_groups[i] = tuple(sorted(list(group)))

    # a list of groups to delete in the end
        # if they are used in merging
        merged_groups = []

    while(is_linkable(ext_groups)):
        # combining groups into pairs
        combined_groups = list(combinations(ext_groups, 2))
        for comb_group in combined_groups:
            comb_1 = comb_group[0]
            comb_2 = comb_group[1]
            # pair-wise matching
            match_group = set()
            if set(comb_1) & set(comb_2) != set():
                match_group = set(comb_1) | set(comb_2)
                merged_groups.extend([comb_1, comb_2])

            ext_groups.append(tuple(match_group))

        # remove empty tuples and already-merged groups
        ext_groups = [group for group in ext_groups
                    if group != () and group not in merged_groups]

    # sort by group sizes
    ext_groups.sort(key=lambda tup: len(tup), reverse=True)
    return ext_groups
