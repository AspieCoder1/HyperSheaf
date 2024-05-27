from abc import abstractmethod

import torch


class HeteroSheafLearner(torch.nn.Module):
    def __init__(self):
        super(HeteroSheafLearner, self).__init__()

    @abstractmethod
    def predict_sheaf(self, node_feats, he_feats, he_index, node_types, he_types):
        raise NotImplementedError

    def forward(self, node_feats, he_feats, he_index, node_types, he_types):
        return self.predict_sheaf(node_feats, he_feats, he_index, node_types, he_types)
