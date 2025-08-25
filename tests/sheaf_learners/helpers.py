import torch
from dataclasses import dataclass


@dataclass
class SheafLearnerInput:
    node_feats: torch.Tensor
    he_feats: torch.Tensor
    he_index: torch.Tensor
    node_types: torch.Tensor
    he_types: torch.Tensor

    @property
    def n_x(self):
        return self.he_index.size(-1)
