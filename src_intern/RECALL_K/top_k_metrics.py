'''
Calculating metrics for top-k recommended search results
'''
from typing import List

def aux_metrics(labels: List[int], scores: List[float], k=5) -> List[int]:
    '''
    Calculates TP, TN, FP, FN for the main metrics
    '''
    # true positive and true negative
    tp, tn, fp, fn = 0, 0, 0, 0

    # sort out the most recommended items
    ls_sorted_pairs = sorted(list(zip(labels, scores)),\
                               reverse=True, key=lambda tup: tup[1])
    # top k recommended items
    for label, _ in ls_sorted_pairs[:k]:
        # if an item was recommended and it's relevant
        tp += label
        # if an item was recommended and it's not relevant
        fp += 1 - label

    # the rest (len - k), not recommended ones
    for label, _ in ls_sorted_pairs[k:]:
        # if an item was not recommended and it's not relevant
        tn += 1 - label
        # if an item was not recommended and it's relevant
        fn += label

    return tp, tn, fp, fn

def recall_at_k(labels: List[int], scores: List[float], k=5) -> float:
    '''
    Recall@k
    '''
    tp, _, _, fn = aux_metrics(labels, scores, k)
    return tp / (tp + fn)


def precision_at_k(labels: List[int], scores: List[float], k=5) -> float:
    '''
    Precision@k
    '''
    tp, _, fp, _ = aux_metrics(labels, scores, k)
    return tp / (tp + fp)


def specificity_at_k(labels: List[int], scores: List[float], k=5) -> float:
    '''
    Specificity@k
    '''
    _, tn, fp, _ = aux_metrics(labels, scores, k)
    return tn / (tn + fp) if (tn + fp) != 0 else 0 # possible if all the labels are relevant


def f1_at_k(labels: List[int], scores: List[float], k=5) -> float:
    '''
    F1@k
    '''
    tp, _, fp, fn = aux_metrics(labels, scores, k)
    return 2*tp / (2*tp + fp + fn) # f1 = 2*prec*rec/prec+rec
