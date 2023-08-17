from typing import Any, Optional, Union

import numpy.typing as npt
from numpy import dtype, ndarray
from numpy.random import Generator, RandomState
from numpy.typing import ArrayLike

class rv_generic:
    def __init__(
        self, seed: Optional[Union[int, Generator, RandomState]] = None
    ) -> None: ...
    def rvs(self, *args: Any, **kwds: Any) -> npt.NDArray[Any]: ...

class rv_continuous(rv_generic):
    def __init__(
        self,
        momtype: Optional[int] = 1,
        a: Optional[float] = None,
        b: Optional[float] = None,
        xtol: Optional[float] = 1e-14,
        badvalue: Optional[float] = None,
        name: Optional[str] = None,
        longname: Optional[str] = None,
        shapes: Optional[str] = None,
        seed: Optional[Union[int, Generator, RandomState]] = None,
    ) -> None: ...
    def ppf(
        self,
        q: ArrayLike,
        *arg: Any,
        **kwargs: Any,
    ) -> ndarray[Any, dtype[Any]]: ...
    def cdf(
        self,
        x: ArrayLike,
        *arg: Any,
        **kwargs: Any,
    ) -> ndarray[Any, dtype[Any]]: ...

expon: rv_continuous
norm: rv_continuous
poisson: rv_continuous
powerlaw: rv_continuous
