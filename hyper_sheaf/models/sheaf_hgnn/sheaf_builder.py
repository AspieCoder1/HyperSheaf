import abc
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_add

from hyper_sheaf.feature_builders.cp_decomp import CPDecompHeFeatBuilder
from hyper_sheaf.feature_builders.input_feats import InputFeatsHeFeatBuilder
from hyper_sheaf.feature_builders.neighbour_aggregation import (
    EquivariantHeFeatBuilder,
    NodeMeanHeFeatBuilder, )
from hyper_sheaf.sheaf_learners.local_concat import LocalConcatSheafLearner
from hyper_sheaf.sheaf_learners.type_concat import TypeConcatSheafLearner
from hyper_sheaf.sheaf_learners.type_ensemble import TypeEnsembleSheafLearner
from hyper_sheaf.utils import utils
from hyper_sheaf.utils.orthogonal import Orthogonal


# helper functions to predict sigma(MLP(x_v || h_e)) varying how thw attributes for hyperedge are computed


class SheafBuilder(nn.Module):
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
            he_feat_type: str = 'var1',
    ):
        super(SheafBuilder, self).__init__()
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
        elif self.he_feat_type == "cp_decomp":
            self.he_feat_builder = CPDecompHeFeatBuilder(
                hidden_channels=hidden_channels, input_norm=self.norm)
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

    @abc.abstractmethod
    def compute_restriction_maps(self, x, e, hyperedge_index, h_sheaf):
        raise NotImplementedError

    def forward(self, x, e, hyperedge_index, node_types, hyperedge_types):
        num_nodes = x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1)  # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1)  # # x d x f -> E x f

        # predict (_ x d) elements
        xs, es = self.compute_node_hyperedge_features(x, e, hyperedge_index)
        h_sheaf = self.predict_sheaf(xs, es, hyperedge_index, node_types,
                                     hyperedge_types)

        return self.compute_restriction_maps(x, e, hyperedge_index, h_sheaf)


# Build the restriction maps for the Diagonal Case
class SheafBuilderDiag(SheafBuilder):
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
        super(SheafBuilderDiag, self).__init__(
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
        elif self.prediction_type == "cp_decomp":
            self.sheaf_lin.reset_parameters()
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()
        else:
            self.sheaf_lin.reset_parameters()

    # this is exclusively for diagonal sheaf
    def compute_restriction_maps(self, x, e, hyperedge_index, h_sheaf):
        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)

        if self.special_head:
            new_head_mask = [1] * (self.d - 1) + [0]
            new_head = [0] * (self.d - 1) + [1]
            h_sheaf = h_sheaf * torch.tensor(
                new_head_mask, device=x.device
            ) + torch.tensor(new_head, device=x.device)

        self.h_sheaf = h_sheaf  # this is stored in self for testing purpose
        h_sheaf_attributes = h_sheaf.reshape(-1)  # (d*K)

        # from a d-dim tensor assoc to every entrence in edge_index
        # create a sparse incidence Nd x Ed

        # We need to modify indices from the NxE matrix
        # to correspond to the large Nd x Ed matrix, but restrict only on the element of the diagonal of each block
        # indices: scalar [i,j] -> block dxd with indices [d*i, d*i+1.. d*i+d-1; d*j, d*j+1 .. d*j+d-1]
        # attributes: reshape h_sheaf

        d_range = (
            torch.arange(self.d, device=x.device).view(1, -1, 1).repeat(2, 1, 1)
        )  # 2xdx1
        hyperedge_index = hyperedge_index.unsqueeze(1)  # 2x1xK
        hyperedge_index = self.d * hyperedge_index + d_range  # 2xdxK
        hyperedge_index = hyperedge_index.permute(0, 2, 1).reshape(2, -1)  # 2x(d*K)
        h_sheaf_index = hyperedge_index

        # the resulting (index, values) pair correspond to the diagonal of each block sub-matrix
        return h_sheaf_index, h_sheaf_attributes


# Build the restriction maps for the General Case
class SheafBuilderGeneral(SheafBuilder):
    def __init__(
            self,
            stalk_dimension: int,
            hidden_channels: int = 64,
            dropout: float = 0.6,
            allset_input_norm: bool = True,
            sheaf_special_head: bool = False,
            sheaf_pred_block: str = "local_concat",
            sheaf_dropout: bool = False,
            sheaf_normtype: Literal[
                "degree_norm", "block_norm", "sym_degree_norm", "sym_block_norm"
            ] = "degree_norm",
            sheaf_act: str = "sigmoid",
            num_node_types: int = 3,
            num_edge_types: int = 6,
            he_feat_type: str = 'var1',
            **_kwargs
    ):
        super(SheafBuilderGeneral, self).__init__(
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
        self.norm_type = sheaf_normtype

    def reset_parameters(self):
        if self.prediction_type == "transformer":
            self.transformer_lin_layer.reset_parameters()
            self.transformer_layer.reset_parameters()
        elif self.prediction_type == "MLP_var3":
            self.general_sheaf_lin.reset_parameters()
            self.general_sheaf_lin2.reset_parameters()
        else:
            self.general_sheaf_lin.reset_parameters()
        if self.prediction_type == "cp_decomp":
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    def compute_restriction_maps(self, x, e, hyperedge_index, h_sheaf):
        if self.sheaf_dropout:
            h_sheaf = F.dropout(
                h_sheaf, p=self.dropout, training=self.training
            )

            # from a d-dim tensor assoc to every entrence in edge_index
            # create a sparse incidence Nd x Ed

            # modify indices to correspond to the big matrix and assign the weights
            # indices: [i,j] -> [d*i, d*i.. d*i+d-1, d*i+d-1; d*j, d*j+1 .. d*j, d*j+1,..d*j+d-1]

        d_range = torch.arange(self.d, device=x.device)
        d_range_edges = d_range.repeat(self.d).view(
            -1, 1
        )  # 0,1..d,0,1..d..   d*d elems
        d_range_nodes = d_range.repeat_interleave(self.d).view(
            -1, 1
        )  # 0,0..0,1,1..1..d,d..d  d*d elems
        hyperedge_index = hyperedge_index.unsqueeze(1)

        hyperedge_index_0 = self.d * hyperedge_index[0] + d_range_nodes
        hyperedge_index_0 = hyperedge_index_0.permute((1, 0)).reshape(1, -1)
        hyperedge_index_1 = self.d * hyperedge_index[1] + d_range_edges
        hyperedge_index_1 = hyperedge_index_1.permute((1, 0)).reshape(1, -1)
        h_sheaf_index = torch.concat((hyperedge_index_0, hyperedge_index_1), 0)
        h_sheaf_attributes = h_sheaf.reshape(-1)

        # create the big matrix from the dxd blocks
        return h_sheaf_index, h_sheaf_attributes


# Build the restriction maps for the Orthogonal Case
class SheafBuilderOrtho(SheafBuilder):
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
            **_kwargs,
    ):
        super(SheafBuilderOrtho, self).__init__(
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

    def reset_parameters(self):
        if self.prediction_type == "MLP_var3":
            self.orth_sheaf_lin.reset_parameters()
            self.orth_sheaf_lin2.reset_parameters()
        else:
            self.orth_sheaf_lin.reset_parameters()
        if self.prediction_type == "cp_decomp":
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    def compute_restriction_maps(self, x, e, hyperedge_index, h_sheaf):
        # convert the d*(d-1)//2 params into orthonormal dxd matrices using housholder transformation
        h_sheaf = self.orth_transform(
            h_sheaf
        )  # sparse version of a NxExdxd tensor

        if self.sheaf_dropout:
            h_sheaf = F.dropout(
                h_sheaf, p=self.dropout, training=self.training
            )

        if self.special_head:
            # add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
            # add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
            new_head_mask = np.ones((self.d, self.d))
            new_head_mask[:, -1] = np.zeros((self.d))
            new_head_mask[-1, :] = np.zeros((self.d))
            new_head = np.zeros((self.d, self.d))
            new_head[-1, -1] = 1
            h_sheaf = h_sheaf * torch.tensor(
                new_head_mask, device=x.device
            ) + torch.tensor(new_head, device=x.device)
            h_sheaf = h_sheaf.float()
        # h_sheaf = h_sheaf * torch.eye(self.d, device=self.device)

        # from a d-dim tensor assoc to every entrence in edge_inde
        # create a sparse incidence Nd x Ed
        # modify indices to correspond to the big matrix and assign the weights
        # indices: [i,j] -> [d*i, d*i.. d*i+d-1, d*i+d-1; d*j, d*j+1 .. d*j, d*j+1,..d*j+d-1]

        d_range = torch.arange(self.d, device=x.device)
        d_range_edges = d_range.repeat(self.d).view(
            -1, 1
        )  # 0,1..d,0,1..d..   d*d elems
        d_range_nodes = d_range.repeat_interleave(self.d).view(
            -1, 1
        )  # 0,0..0,1,1..1..d,d..d  d*d elems
        hyperedge_index = hyperedge_index.unsqueeze(1)

        hyperedge_index_0 = self.d * hyperedge_index[0] + d_range_nodes
        hyperedge_index_0 = hyperedge_index_0.permute((1, 0)).reshape(1, -1)
        hyperedge_index_1 = self.d * hyperedge_index[1] + d_range_edges
        hyperedge_index_1 = hyperedge_index_1.permute((1, 0)).reshape(1, -1)
        h_orth_sheaf_index = torch.concat((hyperedge_index_0, hyperedge_index_1), 0)
        # !!! Is this the correct reshape??? Please check!!
        h_orth_sheaf_attributes = h_sheaf.reshape(-1)

        # create the big matrix from the dxd orthogonal blocks
        return h_orth_sheaf_index, h_orth_sheaf_attributes


# Build the restriction maps for the LowRank Case
class SheafBuilderLowRank(SheafBuilder):
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
            sheaf_normtype: str = "degree_norm",
            rank: int = 2,
            num_node_types: int = 4,
            num_edge_types: int = 6,
            he_feat_type: str = 'var1',
            **_kwargs,
    ):
        super(SheafBuilderLowRank, self).__init__(
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
            self.general_sheaf_lin.reset_parameters()
            self.general_sheaf_lin2.reset_parameters()
        else:
            self.general_sheaf_lin.reset_parameters()
        if self.prediction_type == "cp_decomp":
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    def compute_restriction_maps(self, x, e, hyperedge_index, h_sheaf):
        row, col = hyperedge_index

        # compute AB^T + diag(c)
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
        # add elements on the diagonal
        diag = torch.diag_embed(h_sheaf_C)
        h_sheaf = h_sheaf + diag

        h_sheaf = h_sheaf.reshape(
            h_sheaf.shape[0], self.d * self.d
        )
        if self.sheaf_dropout:
            h_sheaf = F.dropout(
                h_sheaf, p=self.dropout, training=self.training
            )

        # from a d-dim tensor assoc to every entrence in edge_index
        # create a sparse incidence Nd x Ed

        # modify indices to correspond to the big matrix and assign the weights
        # indices: [i,j] -> [d*i, d*i.. d*i+d-1, d*i+d-1; d*j, d*j+1 .. d*j, d*j+1,..d*j+d-1]

        d_range = torch.arange(self.d, device=x.device)
        d_range_edges = d_range.repeat(self.d).view(
            -1, 1
        )  # 0,1..d,0,1..d..   d*d elems
        d_range_nodes = d_range.repeat_interleave(self.d).view(
            -1, 1
        )  # 0,0..0,1,1..1..d,d..d  d*d elems
        hyperedge_index = hyperedge_index.unsqueeze(1)

        hyperedge_index_0 = self.d * hyperedge_index[0] + d_range_nodes
        hyperedge_index_0 = hyperedge_index_0.permute((1, 0)).reshape(1, -1)
        hyperedge_index_1 = self.d * hyperedge_index[1] + d_range_edges
        hyperedge_index_1 = hyperedge_index_1.permute((1, 0)).reshape(1, -1)
        h_sheaf_index = torch.concat((hyperedge_index_0, hyperedge_index_1), 0)

        if self.norm_type == "block_norm":
            # pass
            h_sheaf_1 = h_sheaf.reshape(
                h_sheaf.shape[0], self.d, self.d
            )
            num_nodes = hyperedge_index[0].max().item() + 1
            num_edges = hyperedge_index[1].max().item() + 1

            to_be_inv_nodes = torch.bmm(
                h_sheaf_1, h_sheaf_1.permute(0, 2, 1)
            )
            to_be_inv_nodes = scatter_add(
                to_be_inv_nodes, row, dim=0, dim_size=num_nodes
            )

            to_be_inv_edges = torch.bmm(
                h_sheaf_1.permute(0, 2, 1), h_sheaf_1
            )
            to_be_inv_edges = scatter_add(
                to_be_inv_edges, col, dim=0, dim_size=num_edges
            )

            d_sqrt_inv_nodes = utils.batched_sym_matrix_pow(
                to_be_inv_nodes, -1.0
            )  # n_nodes x d x d
            d_sqrt_inv_edges = utils.batched_sym_matrix_pow(
                to_be_inv_edges, -1.0
            )  # n_edges x d x d

            d_sqrt_inv_nodes_large = torch.index_select(
                d_sqrt_inv_nodes, dim=0, index=row
            )
            d_sqrt_inv_edges_large = torch.index_select(
                d_sqrt_inv_edges, dim=0, index=col
            )

            alpha_norm = torch.bmm(d_sqrt_inv_nodes_large, h_sheaf_1)
            alpha_norm = torch.bmm(alpha_norm, d_sqrt_inv_edges_large)
            h_sheaf = alpha_norm.clamp(min=-1, max=1)
            h_sheaf = h_sheaf.reshape(
                h_sheaf.shape[0], self.d * self.d
            )

        h_sheaf_attributes = h_sheaf.reshape(-1)
        # create the big matrix from the dxd blocks
        return h_sheaf_index, h_sheaf_attributes
