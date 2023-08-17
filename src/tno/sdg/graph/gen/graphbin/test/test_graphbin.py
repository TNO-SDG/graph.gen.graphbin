"""
Tests for the GraphBin class.
"""
# pylint: disable=protected-access
import numpy as np
import pytest

from tno.sdg.graph.gen.graphbin import (
    GraphBinFromScratch,
    TooManyClustersError,
    ZeroValuesError,
)


@pytest.mark.parametrize("n_samples", [10, 100, 1000, 10000])
def test_graphbin_num_nodes(n_samples: int) -> None:
    """
    Test that the number of nodes in the graph is equal to the request number.

    :param n_samples: The number of nodes to sample.
    """
    graphbin = GraphBinFromScratch(n_samples=n_samples)
    synthetic_graph = graphbin._generate_nodes()
    assert len(synthetic_graph.degree) == len(synthetic_graph.index)
    assert len(synthetic_graph.feature) == len(synthetic_graph.index)
    assert len(synthetic_graph.index) == n_samples


def test_graphbin_num_nodes_negative_throws() -> None:
    """
    Test that the binning throws an error when the number of nodes is negative.
    """
    with pytest.raises(ValueError):
        GraphBinFromScratch(n_samples=-1)


def test_graphbin_num_nodes_zero_throws() -> None:
    """
    Test that the binning throws an error when the number of nodes is zero.
    """
    with pytest.raises(ValueError):
        graphbin = GraphBinFromScratch(n_samples=0)
        graphbin.generate()


def test_graphbin_binning_with_features_zeros_throws() -> None:
    """
    Test that the binning throws an error when the features contain zeros.
    """
    graphbin = GraphBinFromScratch(n_samples=100)
    synthetic_graph = graphbin._generate_nodes()
    synthetic_graph.feature[0] = 0

    with pytest.raises(ZeroValuesError):
        graphbin._perform_binning(synthetic_graph, k=5)


def test_graphbin_binning_with_degrees_zeros_throws() -> None:
    """
    Test that the binning throws an error when the degrees contain zeros.
    """
    graphbin = GraphBinFromScratch(n_samples=100)
    synthetic_graph = graphbin._generate_nodes()
    synthetic_graph.degree[0] = 0

    with pytest.raises(ZeroValuesError):
        graphbin._perform_binning(synthetic_graph, k=5)


@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
@pytest.mark.parametrize("k", [2, 4, 8, 16, 32])
def test_graphbin_binning_cluster_amount(n_samples: int, k: int) -> None:
    """
    Test that the number of clusters is equal to the requested number.

    :param n_samples: The number of nodes to sample.
    :param k: The number of clusters to generate.
    """
    graphbin = GraphBinFromScratch(n_samples=n_samples)
    synthetic_graph = graphbin._generate_nodes()
    binning = graphbin._perform_binning(synthetic_graph, k=k)
    assert len(np.unique(binning)) == k


@pytest.mark.parametrize("value", [2, 4, 8])
def test_graphbin_binning_cluster_amount_equal_to_nodes(value: int) -> None:
    """
    Test that the binning does not throw an error when the number of clusters
    is equal to the number of nodes.

    This can create problems with the normalization on certain edge cases,
    specifically if the number of nodes/clusters is small
    (i.e. all features/edges zero/one/equal).

    :param value: Both the number of nodes and the number of clusters.
    """
    for _ in range(100):
        graphbin = GraphBinFromScratch(n_samples=value)
        synthetic_nodes = graphbin._generate_nodes()
        graphbin._perform_binning(synthetic_nodes, k=value)


def test_graphbin_binning_cluster_amount_too_high() -> None:
    """
    Test that the binning throws an error when the number of clusters is too
    high, namely when the number of clusters is larger than the number of
    nodes.
    """
    graphbin = GraphBinFromScratch(n_samples=10)
    synthetic_graph = graphbin._generate_nodes()
    with pytest.raises(TooManyClustersError):
        graphbin._perform_binning(synthetic_graph, k=20)


def test_graphbin_binning_cluster_amount_negative_throws() -> None:
    """
    Test that the binning throws an error when the number of clusters is negative.
    """
    graphbin = GraphBinFromScratch(n_samples=10)
    synthetic_graph = graphbin._generate_nodes()
    with pytest.raises(ValueError):
        graphbin._perform_binning(synthetic_graph, k=-1)


def test_graphbin_binning_cluster_amount_zero_throws() -> None:
    """
    Test that the binning throws an error when the number of clusters is zero.
    """
    graphbin = GraphBinFromScratch(n_samples=10)
    synthetic_graph = graphbin._generate_nodes()
    with pytest.raises(ValueError):
        graphbin._perform_binning(synthetic_graph, k=0)


@pytest.mark.parametrize("n_samples", [10, 50])
def test_graphbin_edge_gen_verify_correctness_of_edges(n_samples: int) -> None:
    """
    Test that the generated edges are valid edges, i.e. that their start and
    end correspond to existing nodes.

    :param n_samples: The number of nodes to sample.
    """
    graphbin = GraphBinFromScratch(n_samples=n_samples)
    graph = graphbin.generate()

    # For n_samples nodes, the node ids range from 0 to n_samples-1
    assert np.all(graph.edges >= 0)
    assert np.all(graph.edges < n_samples)


@pytest.mark.parametrize("n_samples", [10, 50])
def test_graphbin_edge_gen_does_not_contain_nodes_without_edges(n_samples: int) -> None:
    """
    Test that the generated edges do not contain nodes without edges.

    :param n_samples: The number of nodes to sample.
    """
    graphbin = GraphBinFromScratch(n_samples=n_samples)
    graph = graphbin.generate()

    assert len(np.unique(graph.edges.ravel())) == n_samples
