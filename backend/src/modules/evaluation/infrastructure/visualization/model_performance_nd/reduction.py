import numpy as np
from sklearn.decomposition import PCA


def reduce_to_1d(X: np.ndarray) -> tuple[np.ndarray, object | None]:
    """Return (X1d, reducer_or_None). PCA if multi-D, identity if already 1D."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[1] == 1:
        return X[:, 0], None
    reducer = PCA(n_components=1)
    Xr = reducer.fit_transform(X).squeeze()
    return Xr, reducer


def transform_1d(reducer: object | None, X: np.ndarray | None) -> np.ndarray | None:
    if X is None:
        return None
    X = np.asarray(X)
    if reducer is None:
        return X[:, 0] if X.shape[1] == 1 else None
    return reducer.transform(X).squeeze()
