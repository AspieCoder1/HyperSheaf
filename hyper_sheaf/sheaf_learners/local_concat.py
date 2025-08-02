import torch
import torch.nn.functional as F

from hyper_sheaf.utils.mlp import MLP
from .core import HeteroSheafLearner


class LocalConcatSheafLearner(HeteroSheafLearner):
    def __init__(
        self,
        node_feats: int,
        out_channels: int,
        hidden_channels: int = 64,
        norm: bool = True,
        act_fn: str = "relu",
    ):
        super(LocalConcatSheafLearner, self).__init__(act_fn=act_fn)
        self.lin = MLP(
            in_channels=2 * node_feats,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=1,
            dropout=0.0,
            normalisation="ln",
            input_norm=norm,
        )

    def predict_sheaf(self, node_feats, he_feats, he_index, node_types, he_types):
        h_sheaf = torch.cat((node_feats, he_feats), dim=-1)
        h_sheaf = self.lin(h_sheaf)
        return h_sheaf
