from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from hyper_sheaf.hyperedge_feature_builders import (
    compute_hyperedge_features_var1,
    compute_hyperedge_features_var2,
    compute_hyperedge_features_var3,
    compute_hyperedge_index_cp_decomp,
)
from hyper_sheaf.models.mlp import MLP
from hyper_sheaf.sheaf_learners import (
    predict_block_local_concat,
    predict_block_type_concat,
    predict_block_type_ensemble,
)


class HGNCSheafBuilder(nn.Module):
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
        super(HGNCSheafBuilder, self).__init__()
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
            self.cp_W = MLP(
                in_channels=hidden_channels + 1,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
            self.cp_V = MLP(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=hidden_channels,
            )

        self.sheaf_lin = MLP(
            in_channels=2 * self.MLP_hidden,
            hidden_channels=hidden_channels,
            out_channels=self.sheaf_out_channels,
            num_layers=1,
            dropout=0.0,
            normalisation="ln",
            input_norm=self.norm,
        )

        if self.prediction_type == "type_concat":
            self.sheaf_lin = MLP(
                in_channels=2 * self.MLP_hidden + num_node_types + num_edge_types,
                hidden_channels=hidden_channels,
                out_channels=self.sheaf_out_channels,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm, )
        elif self.prediction_type == "type_ensemble":
            self.type_layers = nn.ModuleList([
                MLP(
                    in_channels=2 * self.MLP_hidden,
                    hidden_channels=hidden_channels,
                    out_channels=self.sheaf_out_channels,
                    num_layers=1,
                    dropout=0.0,
                    normalisation="ln",
                    input_norm=self.norm,
                )
                for _ in range(num_edge_types)
            ])

    def compute_node_hyperedge_features(self, x, e, hyperedge_index):
        if self.he_feat_type == 'var1':
            return compute_hyperedge_features_var1(x, e, hyperedge_index)
        elif self.he_feat_type == 'var2':
            return compute_hyperedge_features_var2(x, hyperedge_index)
        elif self.he_feat_type == 'var3':
            return compute_hyperedge_features_var3(x, hyperedge_index, self.sheaf_phi)
        elif self.he_feat_type == 'cp_decomp':
            return compute_hyperedge_index_cp_decomp(x, hyperedge_index, self.cp_W,
                                                     self.cp_V)

    def predict_sheaf(self, xs, es, hyperedge_index, node_types, hyperedge_types):
        if self.prediction_type == "type_concat":
            return predict_block_type_concat(
                xs, es, hyperedge_index, node_types, hyperedge_types, self.sheaf_lin,
                self.sheaf_act
            )
        if self.prediction_type == "type_ensemble":
            return predict_block_type_ensemble(
                xs,
                es,
                hyperedge_index,
                hyperedge_types,
                self.type_layers,
                self.sheaf_act
            )
        return predict_block_local_concat(
            xs, es, self.sheaf_lin, self.sheaf_act
        )


class HGCNSheafBuilderDiag(HGNCSheafBuilder):
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


class HGCNSheafBuilderGeneral(HGNCSheafBuilder):
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


class HGCNSheafBuilderOrtho(HGNCSheafBuilder):
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
