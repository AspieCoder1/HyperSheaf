import torch

from .base_builder import BaseHeFeatBuilder


class InputFeatsHeFeatBuilder(BaseHeFeatBuilder):
    def __init__(self, **_kwargs):
        super().__init__()

    def compute_he_features(self, x, he_feats, hyperedge_index):
        row, col = hyperedge_index
        xs = torch.index_select(x, dim=0, index=row)
        es = torch.index_select(he_feats, dim=0, index=col)

        return xs, es
