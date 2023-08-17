"""
This module contains the Graph class.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


def numpy_factory_int() -> npt.NDArray[np.int_]:
    """
    This is a factory function for numpy arrays, but includes type hints to
    make the static type checker happy.

    :return: An empty numpy array.
    """
    return np.array([])


def numpy_factory_float() -> npt.NDArray[np.float_]:
    """
    This is a factory function for numpy arrays, but includes type hints to
    make the static type checker happy.

    :return: An empty numpy array.
    """
    return np.array([])


@dataclass
class Graph:
    """
    Simple class to represent a Graph structure.
    """

    index: npt.NDArray[np.int_] = field(default_factory=numpy_factory_int)
    """List containing the index of each node."""
    feature: npt.NDArray[np.float_] = field(default_factory=numpy_factory_float)
    """List containing the feature value of each node."""
    edges: npt.NDArray[np.int_] = field(default_factory=numpy_factory_int)
    """List containing all pairs of edges."""

    _degree: npt.NDArray[np.int_] = field(default_factory=numpy_factory_int)
    """Contains a view of the degree of each node."""

    @property
    def degree(self) -> npt.NDArray[np.int_]:
        """
        This property returns a view of the degree of each node.

        :return: A view of the degree of each node.
        """
        return self._degree


@dataclass
class SyntheticNodes:
    """
    This class functions as a limited description of a graph, namely only the
    nodes and their degrees. This description of a graph has no edges.
    """

    index: npt.NDArray[np.int_] = field(default_factory=numpy_factory_int)
    """List containing the index of each node."""
    feature: npt.NDArray[np.float_] = field(default_factory=numpy_factory_float)
    """List containing the feature value of each node."""
    degree: npt.NDArray[np.int_] = field(default_factory=numpy_factory_int)
    """List containing the degree of each node."""
