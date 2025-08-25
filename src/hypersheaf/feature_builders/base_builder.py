import torch
from abc import abstractmethod, ABC


class BaseHeFeatBuilder(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute_he_features(
        self, x, he_feats, hyperedge_index
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(self, x, he_feats, hyperedge_index):
        return self.compute_he_features(x, he_feats, hyperedge_index)

    def __repr__(self):
        return f"{self.__class__.__name__}()"
