#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021
#
# Distributed under terms of the MIT license.

"""
This script contains all models in our paper.
"""
from typing import Literal, Optional

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.dense import Linear
from torch_scatter import scatter_mean

#  This part is for HyperGCN
from .layers import *
from .sheaf_builder import (
    SheafBuilderDiag,
    SheafBuilderOrtho,
    SheafBuilderGeneral,
    SheafBuilderLowRank,
)
from hyper_sheaf.utils.mlp import MLP
from hyper_sheaf.feature_builders import BaseHeFeatBuilder


class SheafHyperGNN(nn.Module):
    """
    This is a Hypergraph Sheaf Model with
    the dxd blocks in H_BIG associated to each pair (node, hyperedge)
    being **diagonal**

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            he_feature_builder: BaseHeFeatBuilder,
            hidden_channels: int = 64,
            sheaf_type: str = "DiagSheafs",
            stalk_dimension: int = 6,
            num_layers: int = 2,
            dropout: float = 0.6,
            sheaf_act: Literal["sigmoid", "tanh", "none"] = "sigmoid",
            init_hedge: Literal["rand", "avg"] = "rand",
            sheaf_normtype: Literal[
                "degree_norm", "block_norm", "sym_degree_norm", "sym_block_norm"
            ] = "degree_norm",
            cuda: int = 0,
            left_proj: bool = False,
            allset_input_norm: bool = True,
            dynamic_sheaf: bool = False,
            residual_connections: bool = False,
            use_lin2: bool = False,
            sheaf_special_head: bool = False,
            sheaf_learner: Literal[
                'local_concat', 'type_concat', 'type_ensemble'] = "local_concat",
            sheaf_dropout: bool = False,
            rank: int = 2,
            is_vshae: bool = False,
            num_node_types: int = 6,
            num_hyperedge_types: int = 3,
            **_kwargs,
    ):
        super(SheafHyperGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout  # Note that default is 0.6
        self.num_features = in_channels
        self.MLP_hidden = hidden_channels
        self.d = stalk_dimension  # dimension of the stalks
        self.init_hedge = (
            init_hedge  # how to initialise hyperedge attributes: avg or rand
        )
        self.norm_type = (
            sheaf_normtype  # type of laplacian normalisation degree_norm or block_norm
        )
        self.act = sheaf_act  # type of nonlinearity used when predicting the dxd blocks
        self.left_proj = left_proj  # multiply with (I x W_1) to the left
        self.norm = allset_input_norm
        self.dynamic_sheaf = (
            dynamic_sheaf  # if True, theb sheaf changes from one layer to another
        )
        self.residual = residual_connections
        self.is_vshae = is_vshae
        self.he_feat_builder = he_feature_builder
        self.pred_block = sheaf_learner

        self.hyperedge_attr = None
        if cuda in [0, 1]:
            self.device = torch.device(
                "cuda:" + str(cuda) if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")

        self.lin = MLP(
            in_channels=self.num_features,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels * self.d,
            num_layers=1,
            dropout=0.0,
            normalisation="ln",
            input_norm=False,
        )
        self.use_lin2 = use_lin2
        self.sheaf_type = sheaf_type

        # define the model and sheaf generator according to the type of sheaf wanted
        # The diuffusion does not change, however tha implementation for diag and ortho is more efficient
        if sheaf_type == "DiagSheafs":
            ModelSheaf, ModelConv = SheafBuilderDiag, HyperDiffusionDiagSheafConv
        elif sheaf_type == "OrthoSheafs":
            ModelSheaf, ModelConv = SheafBuilderOrtho, HyperDiffusionOrthoSheafConv
        elif sheaf_type == "GeneralSheafs":
            ModelSheaf, ModelConv = SheafBuilderGeneral, HyperDiffusionGeneralSheafConv
        else:
            ModelSheaf, ModelConv = SheafBuilderLowRank, HyperDiffusionGeneralSheafConv

        self.convs = nn.ModuleList()
        # Sheaf Diffusion layers
        self.convs.append(
            ModelConv(
                self.MLP_hidden,
                self.MLP_hidden,
                d=self.d,
                device=self.device,
                norm_type=self.norm_type,
                left_proj=self.left_proj,
                norm=self.norm,
                residual=self.residual,
            )
        )

        # Model to generate the reduction maps
        self.sheaf_builder = nn.ModuleList()
        self.sheaf_builder.append(
            ModelSheaf(
                stalk_dimension=stalk_dimension,
                hidden_channels=hidden_channels,
                dropout=dropout,
                allset_input_norm=allset_input_norm,
                sheaf_special_head=sheaf_special_head,
                sheaf_pred_block=sheaf_learner,
                sheaf_dropout=sheaf_dropout,
                sheaf_normtype=self.norm,
                he_feat_builder=he_feature_builder,
                num_edge_types=num_hyperedge_types,
                num_node_types=num_node_types
            )
        )

        self.mu_encoder = ModelConv(
            self.MLP_hidden,
            self.MLP_hidden,
            d=self.d,
            device=self.device,
            norm_type=self.norm_type,
            left_proj=self.left_proj,
            norm=self.norm,
            residual=self.residual,
        )
        self.logstd_encoder = ModelConv(
            self.MLP_hidden,
            self.MLP_hidden,
            d=self.d,
            device=self.device,
            norm_type=self.norm_type,
            left_proj=self.left_proj,
            norm=self.norm,
            residual=self.residual,
        )

        for _ in range(self.num_layers - 1):
            # Sheaf Diffusion layers
            self.convs.append(
                ModelConv(
                    self.MLP_hidden,
                    self.MLP_hidden,
                    d=self.d,
                    device=self.device,
                    norm_type=self.norm_type,
                    left_proj=self.left_proj,
                    norm=self.norm,
                    residual=self.residual,
                )
            )
            # Model to generate the reduction maps if the sheaf changes from one layer to another
            if self.dynamic_sheaf:
                self.sheaf_builder.append(
                    ModelSheaf(
                        stalk_dimension=stalk_dimension,
                        hidden_channels=hidden_channels,
                        dropout=dropout,
                        allset_input_norm=allset_input_norm,
                        sheaf_special_head=sheaf_special_head,
                        sheaf_pred_block=sheaf_learner,
                        sheaf_dropout=sheaf_dropout,
                        sheaf_normtype=self.norm,
                        rank=rank,
                        he_feat_builder=he_feature_builder,
                        num_edge_types=num_hyperedge_types,
                        num_node_types=num_node_types
                    )
                )

        self.out_dim = self.MLP_hidden * self.d
        self.lin2 = Linear(self.MLP_hidden * self.d, out_channels, bias=False)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for sheaf_builder in self.sheaf_builder:
            sheaf_builder.reset_parameters()

        self.lin.reset_parameters()
        self.lin2.reset_parameters()

    def init_hyperedge_attr(self, type, num_edges=None, x=None, hyperedge_index=None):
        # initialize hyperedge attributes either random or as the average of the nodes
        if type == "rand":
            hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        elif type == "avg":
            hyperedge_attr = scatter_mean(
                x[hyperedge_index[0]], hyperedge_index[1], dim=0
            )
        else:
            hyperedge_attr = None
        return hyperedge_attr

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        num_nodes = data.x.shape[0]  # data.edge_index[0].max().item() + 1
        num_edges = data.edge_index[1].max().item() + 1

        # if we are at the first epoch, initialise the attribute, otherwise use the previous ones
        if self.hyperedge_attr is None:
            if hasattr(data, "hyperedge_attr"):
                self.hyperedge_attr = data.hyperedge_attr
            else:
                self.hyperedge_attr = self.init_hyperedge_attr(
                    self.init_hedge,
                    num_edges=num_edges,
                    x=x,
                    hyperedge_index=edge_index,
                )

        # expand the input N x num_features -> Nd x num_features such that we can apply the propagation
        x = self.lin(x)
        x = x.view((x.shape[0] * self.d, self.MLP_hidden))  # (N * d) x num_features

        hyperedge_attr = self.lin(self.hyperedge_attr)
        hyperedge_attr = hyperedge_attr.view(
            (hyperedge_attr.shape[0] * self.d, self.MLP_hidden)
        )

        for i, conv in enumerate(self.convs[:-1]):
            # infer the sheaf as a sparse incidence matrix Nd x Ed, with each block being diagonal
            if i == 0 or self.dynamic_sheaf:
                h_sheaf_index, h_sheaf_attributes = self.sheaf_builder[i](
                    x, hyperedge_attr, edge_index, data.node_types, data.hyperedge_types
                )
            # Sheaf Laplacian Diffusion
            x = F.elu(
                conv(
                    x,
                    hyperedge_index=h_sheaf_index,
                    alpha=h_sheaf_attributes,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                )
            )
            x = F.dropout(x, p=self.dropout, training=self.training)

        # infer the sheaf as a sparse incidence matrix Nd x Ed, with each block being diagonal
        if len(self.convs) == 1 or self.dynamic_sheaf:
            h_sheaf_index, h_sheaf_attributes = self.sheaf_builder[-1](
                x, hyperedge_attr, edge_index
            )
        # Sheaf Laplacian Diffusion
        x = self.convs[-1](
            x,
            hyperedge_index=h_sheaf_index,
            alpha=h_sheaf_attributes,
            num_nodes=num_nodes,
            num_edges=num_edges,
        )
        if self.is_vshae:
            mu = F.elu(
                self.mu_encoder(
                    x,
                    hyperedge_index=h_sheaf_index,
                    alpha=h_sheaf_attributes,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                )
            )
            logstd = F.elu(
                self.logstd_encoder(
                    x,
                    hyperedge_index=h_sheaf_index,
                    alpha=h_sheaf_attributes,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                )
            )
            return mu.view(num_nodes, -1), logstd.view(num_nodes, -1)

        x = x.view(num_nodes, -1)  # Nd x out_channels -> Nx(d*out_channels)
        if self.use_lin2:
            x = self.lin2(x)  # Nx(d*out_channels)-> N x num_classes
        x = F.elu(x)
        return x

    def __repr__(self):
        return f"SheafHyperGNN-{self.sheaf_type}-{self.pred_block}"
