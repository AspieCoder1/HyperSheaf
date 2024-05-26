from torch_geometric.data import Data
from torch_geometric.typing import OptTensor


class HeteroHypergraph(Data):

    def __init__(self, x: OptTensor = None, hyperedge_index: OptTensor = None,
                 node_types: OptTensor = None, hyperedge_types: OptTensor = None,
                 hyperedge_features: OptTensor = None, y: OptTensor = None, ):
        super(HeteroHypergraph, self).__init__(
            x=x,
            edge_index=hyperedge_index,
            y=y,
        )
        self.node_types = node_types
        self.hyperedge_types = hyperedge_types
        self.hyperedge_features = hyperedge_features

    @property
    def num_node_types(self) -> int:
        return int(self.node_types.max()) + 1 if self.node_types is not None else 1

    @property
    def num_hyperedge_types(self) -> int:
        return int(
            self.hyperedge_types.max()) + 1 if self.hyperedge_types is not None else 1

    @property
    def num_hyperedges(self) -> int:
        return self.edge_index[1].max() + 1 if self.edge_index is not None else 0

    @property
    def n_x(self):
        return self.num_nodes

    @property
    def hyperedge_index(self):
        return self.edge_index
