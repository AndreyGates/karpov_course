"""Self-implementation of regression metrics"""
import numpy as np

def mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """MSE"""
    return np.mean(np.square(actual-predicted))

def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """RMSE"""
    return mean_squared_error(actual, predicted) ** 0.5

def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """MAE"""
    return np.mean(np.abs(actual-predicted))

def mean_absolute_percentage_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """MAPE"""
    return np.mean(np.abs(1 - predicted/actual)) * 100

def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Determination coefficient"""
    numerator = np.sum(np.square(actual-predicted))
    denominator = np.sum(np.square(actual-np.mean(actual)))
    rel_variance = numerator / denominator

    return 1 - rel_variance


def test():
    actual = np.array([3, -0.5, 2, 7])
    predicted = np.array([2.5, 0.0, 2, 8])

    assert np.allclose(mean_squared_error(actual, predicted), 0.375)
    assert np.allclose(root_mean_squared_error(actual, predicted), 0.6123724356957945)
    assert np.allclose(mean_absolute_error(actual, predicted), 0.5)
    assert np.allclose(
        mean_absolute_percentage_error(actual, predicted), 32.73809523809524
    )
    assert np.allclose(r_squared(actual, predicted), 0.9486081370449679)

    print("All tests passed.")


if __name__ == "__main__":
    test()
