"""
Exceptions for the GraphBin algorithm.
"""


class GraphBinNotInitializedError(Exception):
    """
    Exception raised when the GraphBin instance has not been initialized with
    a source graph.

    To initialize GraphBin, call `GraphBin.from_scratch(...)`.
    """

    def __init__(
        self,
        message: str = "The GraphBin algorithm is not initialized. Please run "
        "`GraphBin.from_scratch()` to initialize the instance.",
    ):
        self.message = message
        super().__init__(self.message)


class TooManyClustersError(Exception):
    """
    Exception raised when the number of requested clusters is larger than the
    number of nodes in the synthetic graph.
    """

    def __init__(
        self,
        message: str = "The number of nodes in the synthetic graph is smaller "
        "than the number of bins. This is not allowed.",
    ):
        self.message = message
        super().__init__(self.message)


class ZeroValuesError(ValueError):
    """
    Exception raised when an input contains zero values, which the operation
    does not support.
    """

    def __init__(self, message: str = "Value cannot contains zeros."):
        self.message = message
        super().__init__(self.message)
