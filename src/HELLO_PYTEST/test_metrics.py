'''Import tested functions'''
import metrics


def test_profit() -> None:
    '''Test profit'''
    assert metrics.profit([1, 2, 3], [1, 1, 1]) == 3


def test_margin() -> None:
    '''Test margin'''
    assert metrics.margin([1, 2, 3], [1, 1, 1]) == 0.5


def test_markup() -> None:
    '''Test markup'''
    assert metrics.markup([1, 2, 3], [1, 1, 1]) == 1
