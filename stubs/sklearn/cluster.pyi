from typing import Any, Union

from numpy.typing import ArrayLike

class KMeans:
    def __init__(
        self,
        n_clusters: int,
        *args: Any,
        init: str = "k-means++",
        n_init: Union[int, str] = "warn",
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: int = 0,
        random_state: Any = None,
        copy_x: bool = True,
        algorithm: str = "lloyd",
    ) -> None: ...
    def fit(
        self, X: ArrayLike, y: None = None, sample_weight: Union[ArrayLike, None] = None
    ) -> "KMeans": ...
    def predict(self, X: ArrayLike, sample_weight: str = "deprecated") -> ArrayLike: ...
