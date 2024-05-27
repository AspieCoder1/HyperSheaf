import torch
import torch.nn as nn
import torch.nn.functional as F

from hyper_sheaf.utils.mlp import MLP
from .core import HeteroSheafLearner


class TypeEnsembleSheafLearner(HeteroSheafLearner):
    def __init__(self, node_feats: int, out_channels: int, num_he_types,
                 hidden_channels: int = 64,
                 norm: bool = True, act_fn: str = 'relu'):
        super(TypeEnsembleSheafLearner, self).__init__()
        self.type_layers = nn.ModuleList([
            MLP(
                in_channels=2 * node_feats,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=norm,
            )
            for _ in range(num_he_types)
        ])

        self.act_fn = act_fn

    def predict_sheaf(self, node_feats, he_feats, he_index, node_types, he_types):
        node, hyperedge = he_index
        h_cat = torch.cat((node_feats, he_feats), dim=-1)

        hyperedge_types = torch.index_select(he_types, dim=0,
                                             index=hyperedge).to(
            torch.long)

        unique, counts = torch.unique(hyperedge_types, return_counts=True)
        hyperedge_type_idx = torch.argsort(hyperedge_types)
        hyperedge_type_splits = hyperedge_type_idx.split(split_size=counts.tolist())

        results = []

        for i, split in enumerate(hyperedge_type_splits):
            results.append(self.type_layers[i](h_cat[split]))

        stacked_maps = torch.row_stack(results)
        h_sheaf = torch.empty(stacked_maps.shape, device=stacked_maps.device)
        h_sheaf[hyperedge_type_idx] = stacked_maps

        if self.act_fn == 'relu':
            return F.relu(h_sheaf)
        if self.act_fn == 'sigmoid':
            return F.sigmoid(h_sheaf)
        if self.act_fn == 'tanh':
            return F.tanh(h_sheaf)
        if self.act_fn == 'elu':
            return F.elu(h_sheaf)
        return h_sheaf
