from abc import abstractmethod

import torch
import torch.nn.functional as F


class HeteroSheafLearner(torch.nn.Module):
    def __init__(self, act_fn):
        super(HeteroSheafLearner, self).__init__()
        self.act_fn = act_fn

    @abstractmethod
    def predict_sheaf(self, node_feats, he_feats, he_index, node_types, he_types):
        raise NotImplementedError

    def forward(self, node_feats, he_feats, he_index, node_types, he_types):
        return self.sheaf_act(
            self.predict_sheaf(node_feats, he_feats, he_index, node_types, he_types)
        )

    def sheaf_act(self, h_sheaf):
        if self.act_fn == "relu":
            return F.relu(h_sheaf)
        if self.act_fn == "sigmoid":
            return F.sigmoid(h_sheaf)
        if self.act_fn == "tanh":
            return F.tanh(h_sheaf)
        if self.act_fn == "elu":
            return F.elu(h_sheaf)
        return h_sheaf
