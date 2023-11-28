'''Import tested functions'''
import metrics


def test_non_int_clicks():
    '''Test for non-integer number of clicks'''
    try:
        metrics.ctr(1.5, 2)
    except TypeError:
        pass
    else:
        raise AssertionError("Non int clicks not handled")


def test_non_int_views():
    '''Test for non-integer number of views'''
    try:
        metrics.ctr(1, 1.5)
    except TypeError:
        pass
    else:
        raise AssertionError("Non int views not handled")


def test_non_positive_clicks():
    '''Test for non-positive number of clicks'''
    try:
        metrics.ctr(-1, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative clicks not handled")


def test_non_positive_views():
    '''Test for non-positive number of views'''
    try:
        metrics.ctr(-1, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative views not handled")


def test_clicks_greater_than_views():
    '''Test for clicks > views'''
    try:
        metrics.ctr(3, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("views < clicks not handled")

def test_zero_views():
    '''Test for views = 0'''
    try:
        metrics.ctr(0, 0)
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("Zero views not handled")
