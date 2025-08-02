import torch
from torch_scatter import scatter_mean, scatter_add

from .base_builder import BaseHeFeatBuilder
from hyper_sheaf.utils.mlp import MLP


class NodeMeanHeFeatBuilder(BaseHeFeatBuilder):
    def __init__(self, **_kwargs):
        super().__init__()

    def compute_he_features(self, x, he_feats, hyperedge_index):
        row, col = hyperedge_index
        e = scatter_mean(x[row], col, dim=0)

        xs = torch.index_select(x, dim=0, index=row)
        es = torch.index_select(e, dim=0, index=col)

        return xs, es


class EquivariantHeFeatBuilder(BaseHeFeatBuilder):
    def __init__(
        self,
        num_node_feats,
        out_channels,
        hidden_channels,
        input_norm: bool = True,
        **_kwargs,
    ):
        """

        :param num_node_feats: Number of input node features
        :param out_channels: Number of output hyperedge features
        :param hidden_channels: Number of hidden channels for MLP
        :param input_norm: Normalise input or not
        """
        super().__init__()
        self.phi = MLP(
            num_node_feats,
            hidden_channels,
            out_channels,
            num_layers=1,
            dropout=0.0,
            normalisation="ln",
            input_norm=input_norm,
        )

    def compute_he_features(self, x, he_feats, hyperedge_index):
        row, col = hyperedge_index
        x_e = self.phi(x)
        # sum(φ(x_v)
        e = scatter_add(x_e[row], col, dim=0)
        return torch.index_select(x, dim=0, index=row), torch.index_select(
            e, dim=0, index=col
        )
