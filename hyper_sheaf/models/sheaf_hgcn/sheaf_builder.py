from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from hyper_sheaf.feature_builders.cp_decomp import CPDecompHeFeatBuilder
from hyper_sheaf.feature_builders.neighbour_aggregation import EquivariantHeFeatBuilder, NodeMeanHeFeatBuilder
from hyper_sheaf.feature_builders.input_feats import InputFeatsHeFeatBuilder
from hyper_sheaf.hyperedge_feature_builders import (
    compute_hyperedge_features_var1,
    compute_hyperedge_features_var2,
    compute_hyperedge_features_var3,
    compute_hyperedge_index_cp_decomp,
)
from hyper_sheaf.sheaf_learners import (
    predict_block_local_concat,
    predict_block_type_concat,
    predict_block_type_ensemble,
)
from hyper_sheaf.sheaf_learners.local_concat import LocalConcatSheafLearner
from hyper_sheaf.sheaf_learners.type_concat import TypeConcatSheafLearner
from hyper_sheaf.sheaf_learners.type_ensemble import TypeEnsembleSheafLearner
from hyper_sheaf.utils.mlp import MLP
from hyper_sheaf.utils.orthogonal import Orthogonal


class HGCNSheafBuilder(nn.Module):
    def __init__(
            self,
            stalk_dimension: int,
            hidden_channels: int = 64,
            dropout: float = 0.6,
            allset_input_norm: bool = True,
            sheaf_special_head: bool = False,
            sheaf_pred_block: str = "local_concat",
            sheaf_dropout: bool = False,
            sheaf_act: str = "sigmoid",
            num_node_types: int = 3,
            num_edge_types: int = 6,
            sheaf_out_channels: Optional[int] = None,
            he_feat_type: str = 'var1'
    ):
        super(HGCNSheafBuilder, self).__init__()
        self.prediction_type = (
            sheaf_pred_block  # pick the way hyperedge feartures are computed
        )
        self.sheaf_dropout = sheaf_dropout
        self.special_head = sheaf_special_head  # add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
        self.d = stalk_dimension  # stalk dinension
        self.MLP_hidden = hidden_channels
        self.norm = allset_input_norm
        self.dropout = dropout
        self.sheaf_act = sheaf_act
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.he_feat_type = he_feat_type
        if sheaf_out_channels is None:
            self.sheaf_out_channels = stalk_dimension
        else:
            self.sheaf_out_channels = sheaf_out_channels

        if self.he_feat_type == "var3":
            self.he_feat_builder = EquivariantHeFeatBuilder(
                num_node_feats=self.MLP_hidden, out_channels=hidden_channels,
                hidden_channels=hidden_channels, input_norm=self.norm)
            self.sheaf_phi = MLP(
                in_channels=self.MLP_hidden,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
        elif self.he_feat_type == "cp_decomp":
            self.he_feat_builder = CPDecompHeFeatBuilder(
                hidden_channels=hidden_channels, input_norm=self.norm)
            # self.cp_W = MLP(
            #     in_channels=hidden_channels + 1,
            #     hidden_channels=hidden_channels,
            #     out_channels=hidden_channels,
            #     num_layers=1,
            #     dropout=0.0,
            #     normalisation="ln",
            #     input_norm=self.norm,
            # )
            # self.cp_V = MLP(
            #     in_channels=hidden_channels,
            #     hidden_channels=hidden_channels,
            #     out_channels=hidden_channels,
            #     num_layers=1,
            #     dropout=0.0,
            #     normalisation="ln",
            #     input_norm=hidden_channels,
            # )
        elif self.he_feat_type == "var2":
            self.he_feat_builder = NodeMeanHeFeatBuilder()
        else:
            self.he_feat_builder = InputFeatsHeFeatBuilder()

        if self.prediction_type == 'local_concat':
            self.sheaf_predictor = LocalConcatSheafLearner(
                node_feats=self.MLP_hidden,
                hidden_channels=hidden_channels,
                out_channels=self.sheaf_out_channels,
                norm=self.norm,
                act_fn=sheaf_act
            )
        elif self.prediction_type == "type_concat":
            self.sheaf_predictor = TypeConcatSheafLearner(
                node_feats=self.MLP_hidden,
                hidden_channels=hidden_channels,
                out_channels=self.sheaf_out_channels,
                num_node_types=num_node_types,
                num_he_types=num_edge_types,
                act_fn=sheaf_act,
                norm=self.norm
            )
        elif self.prediction_type == "type_ensemble":
            self.sheaf_predictor = TypeEnsembleSheafLearner(
                node_feats=self.MLP_hidden,
                hidden_channels=hidden_channels,
                out_channels=self.sheaf_out_channels,
                act_fn=sheaf_act,
                num_he_types=num_edge_types
            )

    def compute_node_hyperedge_features(self, x, e, hyperedge_index):
        return self.he_feat_builder(x, e, hyperedge_index)

    def predict_sheaf(self, xs, es, hyperedge_index, node_types, hyperedge_types):
        return self.sheaf_predictor(xs, es, hyperedge_index, node_types, hyperedge_types)


class HGCNSheafBuilderDiag(HGCNSheafBuilder):
    def __init__(
            self,
            stalk_dimension: int,
            hidden_channels: int = 64,
            dropout: float = 0.6,
            allset_input_norm: bool = True,
            sheaf_special_head: bool = False,
            sheaf_pred_block: str = 'local_concat',
            sheaf_dropout: bool = False,
            sheaf_act: str = "sigmoid",
            num_node_types: int = 3,
            num_edge_types: int = 6,
            he_feat_type: str = 'var1',
            **_kwargs,
    ):
        """
        hidden_dim overwrite the self.MLP_hidden used in the normal sheaf HNN
        """

        super(HGCNSheafBuilderDiag, self).__init__(
            stalk_dimension=stalk_dimension,
            hidden_channels=hidden_channels,
            dropout=dropout,
            allset_input_norm=allset_input_norm,
            sheaf_pred_block=sheaf_pred_block,
            sheaf_special_head=sheaf_special_head,
            sheaf_dropout=sheaf_dropout,
            sheaf_act=sheaf_act,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            he_feat_type=he_feat_type,
        )

    def reset_parameters(self):
        if self.prediction_type == "MLP_var3":
            self.sheaf_lin.reset_parameters()
            self.sheaf_lin2.reset_parameters()
        else:
            self.sheaf_lin.reset_parameters()
        if self.prediction_type == "cp_decomp":
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    # this is exclusively for diagonal sheaf
    def forward(self, x, e, hyperedge_index, node_types, hyperedge_types):
        """tmp
        x: Nd x f -> N x f
        e: Ed x f -> E x f
        -> (concat) N x E x (d+1)F -> (linear project) N x E x d (the elements on the diagonal of each dxd block)
        -> (reshape) (Nd x Ed) with NxE diagonal blocks of dimension dxd

        """
        num_nodes = x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1

        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1)  # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1)  # # x d x f -> E x f

        xs, es = self.compute_node_hyperedge_features(x, e, hyperedge_index)
        h_sheaf = self.predict_sheaf(xs, es, hyperedge_index, node_types,
                                     hyperedge_types)

        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)

        return h_sheaf


class HGCNSheafBuilderGeneral(HGCNSheafBuilder):
    def __init__(
            self,
            stalk_dimension: int,
            hidden_channels: int = 64,
            dropout: float = 0.6,
            allset_input_norm: bool = True,
            sheaf_special_head: bool = False,
            sheaf_pred_block: str = "local_concat",
            sheaf_dropout: bool = False,
            sheaf_act: str = "sigmoid",
            num_node_types: int = 3,
            num_edge_types: int = 6,
            he_feat_type: str = 'var1',
            **_kwargs
    ):
        super(HGCNSheafBuilderGeneral, self).__init__(
            stalk_dimension=stalk_dimension,
            hidden_channels=hidden_channels,
            dropout=dropout,
            allset_input_norm=allset_input_norm,
            sheaf_pred_block=sheaf_pred_block,
            sheaf_special_head=sheaf_special_head,
            sheaf_dropout=sheaf_dropout,
            sheaf_act=sheaf_act,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            sheaf_out_channels=stalk_dimension * stalk_dimension,
            he_feat_type=he_feat_type
        )

    def reset_parameters(self):
        if self.prediction_type == "MLP_var3":
            self.sheaf_lin.reset_parameters()
            self.sheaf_lin2.reset_parameters()
        else:
            self.sheaf_lin.reset_parameters()
        if self.prediction_type == "cp_decomp":
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    # this is exclusively for diagonal sheaf
    def forward(self, x, e, hyperedge_index, node_types, hyperedge_types):
        """tmp
        x: Nd x f -> N x f
        e: Ed x f -> E x f
        -> (concat) N x E x (d+1)F -> (linear project) N x E x d (the elements on the diagonal of each dxd block)
        -> (reshape) (Nd x Ed) with NxE diagonal blocks of dimension dxd

        """

        num_nodes = x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1)  # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1)  # # x d x f -> E x f

        xs, es = self.compute_node_hyperedge_features(x, e, hyperedge_index)
        h_sheaf = self.predict_sheaf(xs, es, hyperedge_index, node_types,
                                     hyperedge_types)

        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)

        return h_sheaf


class HGCNSheafBuilderOrtho(HGCNSheafBuilder):
    def __init__(
            self,
            stalk_dimension: int,
            hidden_channels: int = 64,
            dropout: float = 0.6,
            allset_input_norm: bool = True,
            sheaf_special_head: bool = False,
            sheaf_pred_block: str = "local_concat",
            sheaf_dropout: bool = False,
            sheaf_act: str = "sigmoid",
            num_node_types: int = 3,
            num_edge_types: int = 6,
            he_feat_type: str = 'var1',
            **_kwargs
    ):
        super(HGCNSheafBuilderOrtho, self).__init__(
            stalk_dimension=stalk_dimension,
            hidden_channels=hidden_channels,
            dropout=dropout,
            allset_input_norm=allset_input_norm,
            sheaf_pred_block=sheaf_pred_block,
            sheaf_special_head=sheaf_special_head,
            sheaf_dropout=sheaf_dropout,
            sheaf_act=sheaf_act,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            sheaf_out_channels=stalk_dimension * (stalk_dimension - 1) // 2,
            he_feat_type=he_feat_type
        )
        self.orth_transform = Orthogonal(
            d=self.d, orthogonal_map="householder"
        )  # method applied to transform params into ortho dxd matrix

    # this is exclusively for diagonal sheaf
    def forward(self, x, e, hyperedge_index, node_types, hyperedge_types):
        """tmp
        x: Nd x f -> N x f
        e: Ed x f -> E x f
        -> (concat) N x E x (d+1)F -> (linear project) N x E x d (the elements on the diagonal of each dxd block)
        -> (reshape) (Nd x Ed) with NxE diagonal blocks of dimension dxd

        """

        num_nodes = x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1)  # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1)  # # x d x f -> E x f

        xs, es = self.compute_node_hyperedge_features(x, e, hyperedge_index)
        h_sheaf = self.predict_sheaf(xs, es, hyperedge_index, node_types,
                                     hyperedge_types)
        h_sheaf = self.orth_transform(h_sheaf)  # sparse version of a NxExdxd tensor
        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)

        return h_sheaf


class HGCNSheafBuilderLowRank(nn.Module):
    def __init__(
            self,
            stalk_dimension: int,
            hidden_channels: int = 64,
            dropout: float = 0.6,
            allset_input_norm: bool = True,
            sheaf_special_head: bool = False,
            sheaf_pred_block: str = "local_concat",
            sheaf_normtype: str = "degree_norm",
            sheaf_dropout: bool = False,
            sheaf_act: str = "sigmoid",
            num_node_types: int = 3,
            num_edge_types: int = 6,
            he_feat_type: str = 'var1',
            rank: int = 2,
            **_kwargs,
    ):
        super(HGCNSheafBuilderLowRank, self).__init__(
            stalk_dimension=stalk_dimension,
            hidden_channels=hidden_channels,
            dropout=dropout,
            allset_input_norm=allset_input_norm,
            sheaf_pred_block=sheaf_pred_block,
            sheaf_special_head=sheaf_special_head,
            sheaf_dropout=sheaf_dropout,
            sheaf_act=sheaf_act,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            sheaf_out_channels=2 * stalk_dimension * rank + stalk_dimension,
            he_feat_type=he_feat_type
        )
        self.rank = rank  # rank for the block matrices
        self.norm_type = sheaf_normtype

    def reset_parameters(self):
        if self.prediction_type == "MLP_var3":
            self.sheaf_lin.reset_parameters()
            self.sheaf_lin2.reset_parameters()
        else:
            self.sheaf_lin.reset_parameters()
        if self.prediction_type == "cp_decomp":
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    # this is exclusively for diagonal sheaf
    def forward(self, x, e, hyperedge_index, node_types, hyperedge_types):
        """tmp
        x: Nd x f -> N x f
        e: Ed x f -> E x f
        -> (concat) N x E x (d+1)F -> (linear project) N x E x d (the elements on the diagonal of each dxd block)
        -> (reshape) (Nd x Ed) with NxE diagonal blocks of dimension dxd

        """

        num_nodes = x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1)  # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1)  # # x d x f -> E x f

        xs, es = self.compute_node_hyperedge_features(x, e, hyperedge_index)
        h_sheaf = self.predict_sheaf(xs, es, hyperedge_index, node_types,
                                     hyperedge_types)

        # h_sheaf is nnz x (2*d*r)
        h_sheaf_A = h_sheaf[:, : self.d * self.rank].reshape(
            h_sheaf.shape[0], self.d, self.rank
        )  # nnz x d x r
        h_sheaf_B = h_sheaf[
                    :, self.d * self.rank: 2 * self.d * self.rank
                    ].reshape(
            h_sheaf.shape[0], self.d, self.rank
        )  # nnz x d x r
        h_sheaf_C = h_sheaf[:, 2 * self.d * self.rank:].reshape(
            h_sheaf.shape[0], self.d
        )  # nnz x d x r

        h_sheaf = torch.bmm(
            h_sheaf_A, h_sheaf_B.transpose(2, 1)
        )  # rank-r matrix

        diag = torch.diag_embed(h_sheaf_C)
        h_sheaf = h_sheaf + diag

        h_sheaf = h_sheaf.reshape(h_sheaf.shape[0], self.d * self.d)

        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)

        return h_sheaf
