import numpy as np
from typing import Callable, Dict

from sklearn.base import BaseEstimator, TransformerMixin


class CustomNormalizer(BaseEstimator, TransformerMixin):
    """
    This class exist only to fit in pipeline format
    """

    def __init__(self, method="z-score"):
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_array = np.array(X)
        return NormalizationTools.normalize(X_array, self.method)


class NormalizationTools:
    @staticmethod
    def l2(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    @staticmethod
    def normalize(matrix: np.ndarray, method: str) -> np.ndarray:
        method_map: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "l2": NormalizationTools.l2,
        }

        if method not in method_map:
            raise ValueError(
                f"Unknown normalization method '{method}', verify config file."
                f"Available methods: {list(method_map.keys())}"
            )

        return method_map[method](matrix)
