from typing import Callable

from numpy.typing import NDArray

ValidationMetricMethod = Callable[[NDArray, NDArray], float]


def mean_squared_error(y_true, y_pred): ...
