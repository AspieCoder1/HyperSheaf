import torch
from abc import abstractmethod
from abc import ABC as AbstractBaseClass


class BaseHeFeatBuilder(torch.nn.Module, AbstractBaseClass):
    ...

    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute_he_features(self, x, he_feats, hyperedge_index):
        raise NotImplementedError

    def forward(self, x, he_feats, hyperedge_index):
        return self.compute_he_features(x, he_feats, hyperedge_index)
