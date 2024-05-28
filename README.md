# HyperSheaf: a sheaf hypergraph library for heterogeneous data

This is the official code for the **HyperSheaf** library developed as part of my
research project for the MPhil in Advanced Computer Science at the University of
Cambridge.

# Project requirements

- torch
- torch_geometric
- pyg-lib
- torch-scatter
- torch-sparse
- torch-cluster
- torch-spline-curve
- torch-householder

# Minimal example

A very simple concrete example of a heterogeneous sheaf hypergraph neural network is
included in `tutorial.py`.

## Representing heterogeneous hypergraphs

Heterogeneous hypergraphs are represented as a `HeteroHypergraph` object and is
illustrated by this code from `tutorial.py`.

```python
import torch
from hyper_sheaf.data import HeteroHypergraph

num_nodes = 10
num_node_types = 2
num_hyperedge_types = 2
features = torch.rand(num_nodes, 64)
he_index = torch.tensor(
  [[0, 1, 2, 0, 1, 3, 4, 1, 2, 4], [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]])
labels = torch.randint(0, 5, (num_nodes,))
hyperedge_types = torch.randint(0, num_hyperedge_types, (3,))
node_types = torch.randint(0, num_node_types, (num_nodes,))
data = HeteroHypergraph(
  x=features,
  hyperedge_index=he_index,
  y=labels,
  node_types=node_types,
  hyperedge_types=hyperedge_types,
)
```

`HeteroHypergraph` has the following parameters:

- `x`: the node feature matrix of n nodes
- `hypergraph_index`: COO representation of a sparse incidence matrix in COO form with a
  shape `(2, )`.
  The element represents the node index and the second element the hypergraph index.
- `y`: class labels for the downstream task.
- `node_types`: a `(1, n)` tensor of node types
- `hyperedge_types`: a `(1, e)` tensor of hyperedge types where e is the number of
  hyperedges.

## Computing hyperedge features

HyperSheaf provides the `BaseHeFeatBuilder` to build custom hyperedge feature builders
by simply implementing the `compute_he_features` which takes in the node features,
hyperedge features and the hyperedge index.
If we wish to simply return the hyperedge features included in the `HeteroData` object
the feature builder would look like the following.

```python
from hyper_sheaf.feature_builders.base_builder import BaseHeFeatBuilder
import torch


class InputFeatsHeFeatBuilder(BaseHeFeatBuilder):
  ...

  def compute_he_features(self, x, he_feats, hyperedge_index):
    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)
    es = torch.index_select(he_feats, dim=0, index=col)

    return xs, es
```

## Heterogeneous sheaf learners

## Supported model architectures

- [x] SheafHyperGNN
- [x] SheafHyperGCN
- [ ] SheafAllSetTransformer
- [ ] SheafAllDeepSet

