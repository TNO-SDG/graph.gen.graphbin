"""
Implementation of the GraphBin algorithm.
"""
from __future__ import annotations

import logging
from typing import cast

import numpy as np
import numpy.typing as npt
from numpy.random import BitGenerator, Generator, SeedSequence
from scipy.stats import expon, norm, poisson, powerlaw
from sklearn.cluster import KMeans

from .exceptions import (
    GraphBinNotInitializedError,
    TooManyClustersError,
    ZeroValuesError,
)
from .graph import Graph, SyntheticNodes
from .utils.math import log_std

logger = logging.getLogger(__name__)


class GraphBin:
    """
    Class that implements the GraphBin algorithm.
    """

    # State
    _random: np.random.Generator
    """
    The random number generator used to generate the synthetic graph.
    """
    # Parameters
    k: int
    """
    The amount of clusters with which to bin the source graph.
    """

    def __init__(
        self,
        source_graph: Graph,
        k: int,
        random: None | int | BitGenerator | SeedSequence | Generator = None,
    ) -> None:
        """
        Initialize the GraphBin algorithm with a source graph and parameters.

        :param source_graph: The original graph from which to generate a
            synthetic graph.
        :param k: The amount of clusters with which to bin the source graph.
        :param random: If given, random generator is initialized with the
            random state. If None, fresh randomness is used.
        """
        self._source_graph = source_graph
        self.k = k
        self._random = np.random.default_rng(seed=random)

    def _generate_nodes(self) -> SyntheticNodes:
        """
        Generate synthetic nodes.

        Note: Not yet implemented.

        :return: Graph object containing the generated nodes.
        """
        raise NotImplementedError()

    def _perform_binning(
        self, graph: Graph | SyntheticNodes, k: int = 5
    ) -> npt.NDArray[np.int32]:
        """
        Perform binning on the nodes in the source graph.

        The nodes are binned based on their degree and feature values.
        The result is a mapping from node ids to the label of the bin to which
        the node belongs.

        :param graph: A graph from which to learn the binning.
        :param k: Number of clusters, defaults to 5.
        :return: Array of bin labels for each node.
        """
        logger.info("Performing binning...")

        if k < 1:
            raise ValueError("k must be larger than 0.")
        if (sample_size := len(graph.index)) < k:
            raise TooManyClustersError()

        if np.any(graph.degree == 0):
            raise ZeroValuesError(
                "The synthetic graph contains nodes with degree 0. "
                "The node standardization does not handle zero values. "
            )
        if np.any(graph.feature == 0):
            raise ZeroValuesError(
                "The synthetic graph contains nodes with feature value of 0. "
                "The node standardization does not handle zero values. "
            )

        # Create array with node information (feature and
        # degree).
        node_info = np.zeros((sample_size, 2))
        node_info[:, 0] = log_std(graph.feature)
        node_info[:, 1] = log_std(graph.degree)

        edge_kmeans = KMeans(
            n_clusters=k,
            init="random",
            n_init=10,
            random_state=self._random.integers(low=0, high=2**32 - 1),
        )
        edge_kmeans.fit(np.array(node_info))

        return cast(npt.NDArray[np.int32], edge_kmeans.predict(np.array(node_info)))

    def _derive_adjacency_matrix(
        self, edge_binning: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float_]:
        """
        Get the directed, binned adjacency matrix from a set of edges. A cell
        in the `i`'th row and `j`'th column indicates the probability that
        a node of type `i` has an edge to a node of type `j`. This matrix is
        formed by counting the number of edges that a node of type `i` to
        a node of type `j` in edge_binning.

        Note: Not yet implemented.

        :param edge_binning: The output of `GraphBin._perform_binning`, mapping
            each node to a bin in the edges.
        """
        raise NotImplementedError()

    def _generate_edges(
        self,
        synthetic_nodes: SyntheticNodes,
        edge_binning: npt.NDArray[np.int_],
        adj_directed: npt.NDArray[np.float_],
    ) -> Graph:
        """
        Generate directed edges for a set of nodes.

        Using the completely synthetic nodes, create undirected edges between
        these nodes, conditioned on the features a specified adjacency matrix.
        We start by creating stubs (as many as the degree for every node) and
        iteratively connect these stubs to one another, using the probabilities
        that these types of nodes would connect.

        The given graph is mutated. The edges generated are
        undirected. The nodes of the graph are filtered such that only nodes
        that have at least one edge are included in the synthetic graph. The
        edges of the resulting graph may have minor deviations from the
        synthetic degree sampled by the MPSN.

        :param synthetic_graph: A graph with synthetic nodes generated by
            `GraphBin._generate_nodes`. The `Graph._degree` property must also
            be set, indicating how many edges to generate for each node.
        :param edge_binning: The output of `GraphBin._perform_binning`, mapping
            each node to a bin in the edges.
        :param adj_directed: Binned adjacency matrix.
        """
        logger.info("Generating edges...")

        adj: npt.NDArray[np.float_] = adj_directed + np.transpose(adj_directed)
        np.fill_diagonal(adj, np.diag(adj_directed))

        # The following numpy arrays store information about the stubs.
        # Stores the "node id" of a stub, such that the i'th stub has "node id" `stub_node[i]`
        stub_node: npt.NDArray[np.int_]
        # Whether a stub has been connected to another stub
        stub_connected: npt.NDArray[np.bool_]
        # The bin to which a stub belongs
        stub_bin: npt.NDArray[np.int_]

        # Generate the free stubs and initialize their state
        stub_node = np.repeat(synthetic_nodes.index, synthetic_nodes.degree)
        np.random.shuffle(stub_node)
        stub_connected = np.zeros((stub_node.shape[0]), dtype=bool)
        stub_bin = edge_binning[stub_node]

        # Result bookkeeping
        synthetic_edges: list[npt.NDArray[np.int_]] = []

        for i, node in enumerate(stub_node):
            # Don't connect stubs that have already been connected.
            if stub_connected[i]:
                continue

            stub_connected[i] = True
            # If there are no more stubs to connect, stop.
            # A single stub cannot form an edge.
            if np.all(stub_connected):
                break

            # Retrieve the bin to which this stub belongs.
            stub_i_bin = stub_bin[i]
            # Extract the probabilities of this stub connecting to other bins
            # from the adjecency matrix.
            probs_bin_i: npt.NDArray[np.float_] = adj[stub_i_bin]
            probs_bin_i = probs_bin_i / np.sum(probs_bin_i)
            # Find all remaining bins with unconnected stubs
            mask_not_connected = np.logical_not(stub_connected)
            remaining_bins = np.unique(stub_bin[mask_not_connected])
            # Narrow the bin probabilities according to the remaining bins.
            remaining_bin_probs: npt.NDArray[np.float_]
            remaining_bin_probs = probs_bin_i.ravel()[remaining_bins]
            # Sample a bin, given the probabilities.
            sampled_bin: int
            if np.sum(remaining_bin_probs) == 0 or np.isnan(remaining_bin_probs).all():
                # If none of them should be sampled, ignore the probabilities.
                sampled_bin = np.random.choice(remaining_bins, size=1).item()
            else:
                # Randomly sample a bin, given the probabilities.
                remaining_bin_probs = remaining_bin_probs / np.sum(remaining_bin_probs)
                sampled_bin = np.random.choice(
                    remaining_bins, size=1, p=remaining_bin_probs
                ).item()

            # Finally, we sample a free stub from the sampled bin and connect it to stub i.
            sampled_stub_id = np.random.choice(
                np.where(
                    (stub_bin == sampled_bin) & (~stub_connected),
                )[0],
                size=1,
            ).item()
            synthetic_edges.append(np.array([node, stub_node[sampled_stub_id]]))
            # Then we need to indicate that this stub is now taken.
            stub_connected[sampled_stub_id] = True

        # From the generated edges, we extract the nodes that belong to at
        # least one edge, and the counts of edges per node.
        chosen_nodes, sampled_degrees = np.unique(
            np.array(synthetic_edges).reshape(-1), return_counts=True
        )

        return Graph(
            index=chosen_nodes,
            feature=synthetic_nodes.feature[
                np.isin(synthetic_nodes.index, chosen_nodes)
            ],
            edges=np.array(synthetic_edges),
            _degree=sampled_degrees,
        )

    def generate(self) -> Graph:
        """
        Generate a synthetic graph. The edges are generated using the GraphBin algorithm.

        Note: you must first initialize the GraphBin object, either by
        generating a random source graph from scratch (see
        `GraphBinFromScratch`) or by providing one (not yet implemented).

        :raise GraphBinNotInitializedError: If the GraphBin object has not been
            initialized.
        :return: Graph object representing the generated graph.
        """
        if not self._source_graph or self._source_graph.index.size == 0:
            raise GraphBinNotInitializedError()

        synthetic_nodes = self._generate_nodes()
        edge_binning = self._perform_binning(self._source_graph, self.k).astype(np.int_)
        adj_directed = self._derive_adjacency_matrix(edge_binning)
        synthetic_graph = self._generate_edges(
            synthetic_nodes=synthetic_nodes,
            edge_binning=edge_binning,
            adj_directed=adj_directed,
        )

        return synthetic_graph


class GraphBinFromScratch(GraphBin):
    """
    This class is a version of the GraphBin algorithm with some parts
    overwritten to work without a real source graph.

    As a starting point, we use the parameters to generate a random graph,
    which is used both as source graph and as starting point of the synthetic
    graph.

    Since our randomly generated graph does not have real edges, we randomly
    generate an adjacency matrix.
    """

    def __init__(
        self,
        n_samples: int = 100,
        param_feature: int = 28000,
        param_degree: int = 50,
        cor: float = 0.5,
        k: int = 5,
        param_edges: int = 4,
        random_state: None | int | BitGenerator | SeedSequence | Generator = None,
    ) -> None:
        """
        Initialize the GraphBin algorithm for generating a random synthetic
        graph from scratch.

        To generate the random graph, the various parameters are used in the
        following way.

        First, an amount of `sample_size` two-dimensional values is sampled
        from a bi-variate standard normal distribution. A bi-variate
        distribution is used to impose a correlation between the two variables
        (feature, degree) which will be derived next. The cumulative density is
        then used to derive two different variables. The first sampled
        dimension is transformed to an exponential distribution from which the
        first variable is sampled, the shape of which can be changed using
        `param_feature` parameter. This first variable is a node-level domain
        feature of the synthetic graph, such as the feature or transaction
        activity of the node, often referred to as 'feature'. The second
        sampled dimension is transformed to a powerlaw distribution from which
        the second variable is sampled, which can be changed using the
        `param_degree` parameter. This second variable corresponds to the
        degree of the node (number of edges).

        :param n_samples: Number of nodes.
        :param param_feature: Governs the feature distribution.
            This is the parameter used to specify the shape of the exponential
            distribution from which the feature values are sampled
            (`scipy.stats.expon`).
        :param param_degree: Governs the degree distribution. This is the
            parameter used to specify the shape of the powerlaw distribution
            from which the degrees are sampled (`scipy.stats.powerlaw`).
        :param cor: Strength of correlation between transformed feature and
            degree. This is the correlation used to generate the two bi-variate
            normally distributed values.
        :param k: Number of clusters (bins).
        :param param_edges: Roughly relates to the strength of effect of the
            bins on the edge probabilities. A matrix is filled with
            probabilities of edges occurring between different types of nodes
            by sampling from a Poisson distribution and randomly filling in
            these values in the matrix.
        :param random_state: If int, the random generator is seeded with the
            integer. If RandomState instance, the random generator is
            initialized with the random state. If None, fresh randomness is
            used.
        :raise ValueError: If `n_samples` is smaller than 1.
        """
        self.n_samples = n_samples
        self.param_feature = param_feature
        self.param_degree = param_degree
        self.cor = cor
        self.param_edges = param_edges

        if self.n_samples < 1:
            raise ValueError("n_samples must be larger than 0.")

        super().__init__(source_graph=Graph(), k=k, random=random_state)

    def _generate_nodes(self) -> SyntheticNodes:
        """
        Generate random synthetic nodes. See `GraphBinFromScratch.__init__` for
        more information.

        :return: Graph object containing the generated nodes. Each node has
            a "feature" value and a "degree" value.
        """
        logger.info("Generating nodes...")

        # First create normally distributed value to have control over the
        # correlation between the degree and feature.
        # 1. Specify mean and covariance matrix
        mean = np.repeat(0, 2)
        cov = np.reshape(np.repeat(np.repeat(self.cor, 2), 2), (2, 2))
        np.fill_diagonal(cov, 1)
        # 2. Sample nodes from bivariate normal distribution
        sampled_nodes = self._random.multivariate_normal(mean, cov, self.n_samples)
        cdf_values = norm.cdf(sampled_nodes)

        powerlaw_dist = powerlaw.ppf(
            cdf_values[:, 0], scale=self.param_degree, loc=1, a=0.25
        )
        exponential_dist = expon.ppf(cdf_values[:, 1], scale=self.param_feature, loc=1)

        degree = np.array(powerlaw_dist, dtype=int)
        feature = np.round(exponential_dist, 0)
        ids = np.arange(0, self.n_samples)

        return SyntheticNodes(
            index=ids,
            feature=feature,
            degree=degree,
        )

    def _get_adjacency_matrix(self) -> npt.NDArray[np.float_]:
        """
        Generate a directed adjacency matrix by sampling integers from a
        Poisson distribution. The resulting matrix is of the form k by k. A cell
        in the ith row and jth column indicates the probability that a node of type
        i has an edge to a node of type j. The probabilities in the resulting
        graph can differ from these probabilities, because we are restricted
        by the node degrees during edge generation.

        :return: directed (probability) adjacency matrix.
        """
        # We create a random adjacency matrix, since we have no graph to derive it from.
        adj_raw = np.array(poisson.rvs(self.param_edges, size=self.k**2)).reshape(
            self.k, self.k
        )
        adj: npt.NDArray[np.float_] = adj_raw / np.sum(adj_raw)

        return adj

    def generate(self) -> Graph:
        """
        Generate a synthetic graph. The edges are generated using the GraphBin algorithm.

        :return: Graph object representing the generated graph.
        """
        synthetic_nodes = self._generate_nodes()
        # Note here that we have no source graph, and thus perform the
        # clustering on the randomly generated synthetic graph directly.
        edge_binning = self._perform_binning(synthetic_nodes, self.k).astype(np.int_)
        adj_directed = self._get_adjacency_matrix()
        synthetic_graph = self._generate_edges(
            synthetic_nodes=synthetic_nodes,
            edge_binning=edge_binning,
            adj_directed=adj_directed,
        )

        return synthetic_graph
