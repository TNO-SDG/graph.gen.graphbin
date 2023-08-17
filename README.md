# TNO PET Lab - Synthetic Data Generation (SDG) - Graph - Generation - GraphBin

The TNO PET Lab consists of generic software components, procedures, and
functionalities developed and maintained on a regular basis to facilitate and
aid in the development of PET solutions. The lab is a cross-project initiative
allowing us to integrate and reuse previously developed PET functionalities to
boost the development of new protocols and solutions.

The package `tno.sdg.graph.gen.graphbin` is part of the TNO Python Toolbox.

The research activities that led to this protocol and implementation were
supported by TNO's Appl.AI programme.

_Limitations in (end-)use: the content of this software package may solely be
used for applications that comply with international export control laws._  
_This implementation of software has not been audited. Use at your own risk._

## Documentation

Documentation of the `tno.sdg.graph.gen.graphbin` package can be found
[here](https://docs.pet.tno.nl/sdg/graph/gen/graphbin/0.1.1).

## Install

Easily install the `tno.sdg.graph.gen.graphbin` package using pip:

```console
$ python -m pip install tno.sdg.graph.gen.graphbin
```

The package has two groups of optional dependencies:

- `tests`: Required packages for running the tests included in this package
- `scripts`: The packages required to run the example script

## Usage

This repository implements part of the GraphBin algorithm. Currently, the edge
generation step of GraphBin is implemented, but not the node generation. It is
only supported to generate synthetic graphs "from scratch", i.e. without a
source graph from which characteristics are learned. Instead, the current
implementation provides the method `GraphBin.from_scratch`, which generates a
new random graph based on the provided parameters.

The parameters are as follows:

- `n_samples`: The number of nodes to generate
- `param_feature`: Parameter governing exponential distribution from which the
  value of the "feature" is sampled (i.e. transaction amount)
- `param_degree`: Parameter governing the powerlaw distribution from which the
  degrees of the nodes are sampled
- `cor`: Specify the correlation between `param_feature` and `param_degree`
- `param_edges`: Roughly related to the strength of the binning on the edge
  probabilities

Below, examples of feature and degree distributions are shown for different
values of `param_feature` and `param_degree`.

![Graph depicting the exponential distribution for various parameters used to sample feature values](https://raw.githubusercontent.com/TNO-SDG/graph.gen.graphbin/main/figures/param_feature.png)
![Graph depicting the powerlaw distribution for various parameters used to sample the degree amounts](https://raw.githubusercontent.com/TNO-SDG/graph.gen.graphbin/main/figures/param_feature.png)

### Example Script

Be sure to install the `scripts` optional dependency group (see installation
instructions).

```python
import matplotlib.pyplot as plt
import networkx as nx

from tno.sdg.graph.gen.graphbin import GraphBin

N = 200

graphbin = GraphBin.from_scratch(
    n_samples=N,
    param_feature=2000,
    param_degree=19,
    cor=0.3,
    param_edges=4000,
    random_state=80,
)
graph = graphbin.generate()
```

Plot the node degree & node feature.

```python
plt.figure(figsize=(15, 10), dpi=300)
plt.scatter(graph.degree, graph.feature, s=150, alpha=0.65)
plt.xlabel("Node degree")
plt.ylabel("Node feature")
plt.title("Node degree and node feature (node-level feature), for " + str(N) + " nodes")
plt.show()
```

![Graph showing the distribution of the degree of the nodes and the feature of the nodes](https://raw.githubusercontent.com/TNO-SDG/graph.gen.graphbin/main/figures/node_degree_and_feature_example.png)

And the graph:

```python
plt.figure(figsize=(15, 10), dpi=300)
G = nx.Graph()
G.add_nodes_from(graph.index)
G.add_edges_from(tuple(map(tuple, graph.edges)))

pos = nx.spring_layout(G, k=100 / N)
nx.draw(G, node_size=350, node_color=graph.feature, pos=pos)
plt.title("Synthetic graph with nodes colored by feature value")
plt.show()
```

![Graph showing the synthetic graph resulting from the example script](https://raw.githubusercontent.com/TNO-SDG/graph.gen.graphbin/main/figures/synthetic_graph_example.png)
