"""
Implementation of the GraphBin algorithm for generating Synthetic Graphs.
"""

# Explicit re-export of all functionalities, such that they can be imported properly. Following
# https://www.python.org/dev/peps/pep-0484/#stub-files and
# https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-no-implicit-reexport

from .exceptions import TooManyClustersError as TooManyClustersError
from .exceptions import ZeroValuesError as ZeroValuesError
from .graph import Graph as Graph
from .graphbin import GraphBin as GraphBin
from .graphbin import GraphBinFromScratch as GraphBinFromScratch

__version__ = "0.1.1"
