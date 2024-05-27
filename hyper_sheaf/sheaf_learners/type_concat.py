import torch
import torch.nn.functional as F

from hyper_sheaf.utils.mlp import MLP
from .core import HeteroSheafLearner


class TypeConcatSheafLearner(HeteroSheafLearner):
    def __init__(self, node_feats: int, out_channels: int, num_node_types, num_he_types,
                 hidden_channels: int = 64,
                 norm: bool = True, act_fn: str = 'relu'):
        super(TypeConcatSheafLearner, self).__init__(act_fn=act_fn)

        self.lin = MLP(
            in_channels=2 * node_feats + num_node_types + num_he_types,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=1,
            dropout=0.0,
            normalisation="ln",
            input_norm=norm,
        )

    def predict_sheaf(self, node_feats, he_feats, he_index, node_types, he_types):
        node, hyperedge = he_index
        node_types_onehot = F.one_hot(node_types.to(torch.long))
        hyperedge_types_onehot = F.one_hot(he_types.to(torch.long))
        x_type = torch.index_select(node_types_onehot, dim=0, index=node)
        e_type = torch.index_select(hyperedge_types_onehot, dim=0, index=hyperedge)

        # sigma(MLP(x_v || h_e || t_v || t_u)))
        h_sheaf = torch.cat((node_feats, he_feats, x_type, e_type), dim=-1)
        h_sheaf = self.lin(h_sheaf)
        return h_sheaf
