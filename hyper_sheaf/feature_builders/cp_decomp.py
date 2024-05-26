import torch
from torch_scatter import scatter_mul, scatter_add

from .base_builder import BaseHeFeatBuilder
from hyper_sheaf.utils.mlp import MLP


class CPDecompHeFeatBuilder(BaseHeFeatBuilder):
    def __init__(self, hidden_channels, norm: bool = True, **_kwargs):
        super().__init__()
        self.cp_W = MLP(
            in_channels=hidden_channels + 1,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=1,
            dropout=0.0,
            normalisation="ln",
            input_norm=norm,
        )
        self.cp_V = MLP(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=1,
            dropout=0.0,
            normalisation="ln",
            input_norm=norm,
        )

    def compute_he_features(self, x, he_feats, hyperedge_index):
        row, col = hyperedge_index
        xs = torch.index_select(x, dim=0, index=row)

        xs_ones = torch.cat(
            (xs, torch.ones(xs.shape[0], 1).to(xs.device)), dim=-1
        )  # nnz x f+1
        xs_ones_proj = torch.tanh(self.cp_W(xs_ones))  # nnz x r
        xs_prod = scatter_mul(xs_ones_proj, col, dim=0)  # edges x r
        e = torch.relu(self.cp_V(xs_prod))  # edges x f
        e = e + torch.relu(scatter_add(x[row], col, dim=0))
        es = torch.index_select(e, dim=0, index=col)

        return xs, es
