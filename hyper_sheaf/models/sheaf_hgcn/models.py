from typing import Literal

import torch
import torch_sparse
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import Linear
from torch_scatter import scatter_mean, scatter_add

from hyper_sheaf.data import HeteroHypergraph
from hyper_sheaf.models.sheaf_hgcn.hgcn_sheaf_laplacians import (SheafLaplacianDiag,
                                                                 SheafLaplacianOrtho,
                                                                 SheafLaplacianGeneral, )
from hyper_sheaf.models.sheaf_hgcn.sheaf_builder import (HGCNSheafBuilderDiag,
                                                         HGCNSheafBuilderOrtho,
                                                         HGCNSheafBuilderGeneral,
                                                         HGCNSheafBuilderLowRank, )
from hyper_sheaf.utils import utils
from hyper_sheaf.utils.mlp import MLP


class SheafHyperGCN(nn.Module):
    # replace hyperedge with edges amax(F_v<e(x_v)) ~ amin(F_v<e(x_v))
    def __init__(
        self,
        V,
        in_channels,
        out_channels,
        num_layers: int = 2,
        sheaf_type: str = "DiagSheafs",
        cuda: int = 0,
        hidden_channels: int = 64,
        stalk_dimension: int = 6,
        dropout: float = 0.6,
        sheaf_act: Literal["sigmoid", "tanh", "none"] = "sigmoid",
        init_hedge: Literal["rand", "avg"] = "rand",
        sheaf_normtype: Literal[
            "degree_norm", "block_norm", "sym_degree_norm", "sym_block_norm"
        ] = "degree_norm",
        left_proj: bool = False,
        allset_input_norm: bool = True,
        dynamic_sheaf: bool = False,
        residual_connections: bool = False,
        use_lin2: bool = False,
        sheaf_special_head: bool = False,
        sheaf_learner: Literal[
            'Sheaf-NSD', 'Sheaf-TE', 'Sheaf-ensemble'] = "Sheaf-NSD",
        he_feat_type: Literal['var1', 'var2', 'var3', 'cp_decomp'] = 'var1',
        sheaf_dropout: bool = False,
        rank: int = 2,
        num_node_types: int = 6,
        num_hyperedge_types: int = 3,
        mediators: bool = False,
        **_kwargs,
    ):
        super(SheafHyperGCN, self).__init__()
        d, l, c = in_channels, num_layers, out_channels

        self.num_nodes = V
        h = [hidden_channels]
        for i in range(l - 1):
            power = l - i + 2
            h.append(2**power)
        h.append(c)

        reapproximate = False  # for HyperGCN we take care of this via dynamic_sheaf

        self.MLP_hidden = hidden_channels
        self.d = stalk_dimension

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
        self.sheaf_type = (
            sheaf_type  #'DiagSheafs', 'OrthoSheafs', 'GeneralSheafs' or 'LowRankSheafs'
        )

        self.hyperedge_attr = None
        self.residual = residual_connections

        if cuda in [0, 1]:
            self.device = torch.device(
                "cuda:" + str(cuda) if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")

        # sheaf_type = 'OrthoSheafs'
        if sheaf_type == "DiagSheafs":
            ModelSheaf, self.Laplacian = HGCNSheafBuilderDiag, SheafLaplacianDiag
        elif sheaf_type == "OrthoSheafs":
            ModelSheaf, self.Laplacian = HGCNSheafBuilderOrtho, SheafLaplacianOrtho
        elif sheaf_type == "GeneralSheafs":
            ModelSheaf, self.Laplacian = HGCNSheafBuilderGeneral, SheafLaplacianGeneral
        else:
            ModelSheaf, self.Laplacian = HGCNSheafBuilderLowRank, SheafLaplacianGeneral

        if self.left_proj:
            self.lin_left_proj = nn.ModuleList(
                [
                    MLP(
                        in_channels=self.d,
                        hidden_channels=self.d,
                        out_channels=self.d,
                        num_layers=1,
                        dropout=0.0,
                        normalisation="ln",
                        input_norm=self.norm,
                    )
                    for i in range(l)
                ]
            )

        self.lin = MLP(
            in_channels=self.num_features,
            hidden_channels=self.MLP_hidden,
            out_channels=self.MLP_hidden * self.d,
            num_layers=1,
            dropout=0.0,
            normalisation="ln",
            input_norm=False,
        )

        self.sheaf_builder = nn.ModuleList()
        self.sheaf_builder.append(
            ModelSheaf(
                stalk_dimension=stalk_dimension,
                hidden_channels=hidden_channels,
                dropout=dropout,
                allset_input_norm=allset_input_norm,
                sheaf_special_head=sheaf_special_head,
                sheaf_learner=sheaf_learner,
                sheaf_dropout=sheaf_dropout,
                rank=rank,
                he_feat_type=he_feat_type,
                num_node_types=num_node_types,
                num_edge_types=num_hyperedge_types
            )
        )

        self.out_dim = h[-1] * self.d
        self.lin2 = Linear(self.out_dim, out_channels, bias=False)
        self.use_lin2 = use_lin2

        if self.dynamic_sheaf:
            for i in range(1, l):
                self.sheaf_builder.append(
                    ModelSheaf(
                        stalk_dimension=stalk_dimension,
                        hidden_channels=h[i],
                        dropout=dropout,
                        allset_input_norm=allset_input_norm,
                        sheaf_special_head=sheaf_special_head,
                        sheaf_learner=sheaf_learner,
                        sheaf_dropout=sheaf_dropout,
                        he_feat_type=he_feat_type,
                        num_edge_types=num_hyperedge_types,
                        num_node_types=num_node_types
                    )
                )

        self.layers = nn.ModuleList(
            [
                utils.HyperGraphConvolution(h[i], h[i + 1], reapproximate, cuda)
                for i in range(l)
            ]
        )
        self.do, self.l = dropout, num_layers
        self.m = mediators

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.left_proj:
            for lin_layer in self.lin_left_proj:
                lin_layer.reset_parameters()
        self.lin.reset_parameters()
        self.lin2.reset_parameters()
        for sheaf_builder in self.sheaf_builder:
            sheaf_builder.reset_parameters()

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

    def normalise(self, A, hyperedge_index, num_nodes, d):
        if self.norm_type == "degree_norm":
            # compute D^-1
            D = scatter_add(
                hyperedge_index.new_ones(hyperedge_index.size(1)),
                hyperedge_index[0],
                dim=0,
                dim_size=num_nodes * d,
            )
            D = torch.pow(D, -1.0)
            D[D == float("inf")] = 0
            D = utils.sparse_diagonal(D, (D.shape[0], D.shape[0]))
            D = D.coalesce()

            # compute D^-1A
            A = torch_sparse.spspmm(
                D.indices(),
                D.values(),
                A.indices(),
                A.values(),
                D.shape[0],
                D.shape[1],
                A.shape[1],
            )
            A = torch.sparse_coo_tensor(
                A[0], A[1], size=(num_nodes * d, num_nodes * d)
            ).to(D.device)
        elif self.norm_type == "sym_degree_norm":
            # compute D^-0.5
            D = scatter_add(
                hyperedge_index.new_ones(hyperedge_index.size(1)),
                hyperedge_index[0],
                dim=0,
                dim_size=num_nodes * d,
            )
            D = torch.pow(D, -0.5)
            D[D == float("inf")] = 0
            D = utils.sparse_diagonal(D, (D.shape[0], D.shape[0]))
            D = D.coalesce()

            # compute D^-0.5AD^-0.5
            A = torch_sparse.spspmm(
                D.indices(),
                D.values(),
                A.indices(),
                A.values(),
                D.shape[0],
                D.shape[1],
                A.shape[1],
                coalesced=True,
            )
            A = torch_sparse.spspmm(
                A[0],
                A[1],
                D.indices(),
                D.values(),
                D.shape[0],
                D.shape[1],
                D.shape[1],
                coalesced=True,
            )
            A = torch.sparse_coo_tensor(
                A[0], A[1], size=(num_nodes * d, num_nodes * d)
            ).to(D.device)

        elif self.norm_type == "block_norm":
            # D computed based on the block diagonal
            D = A.to_dense().view((num_nodes, d, num_nodes, d))
            D = torch.permute(D, (0, 2, 1, 3))  # num_nodes x num_nodes x d x d
            D = torch.diagonal(
                D, dim1=0, dim2=1
            )  # d x d x num_nodes (the block diagonal ones)
            D = torch.permute(D, (2, 0, 1))  # num_nodes x d x d

            if self.sheaf_type in ["GeneralSheafs", "LowRankSheafs"]:
                D = utils.batched_sym_matrix_pow(D, -1.0)  # num_nodes x d x d
            else:
                D = torch.pow(D, -1.0)
                D[D == float("inf")] = 0
            D = torch.block_diag(*torch.unbind(D, 0))
            D = D.to_sparse()

            # compute D^-1A
            A = torch.sparse.mm(D, A)  # this is laplacian delta
            if self.sheaf_type in ["GeneralSheafs", "LowRankSheafs"]:
                A = A.to_dense().clamp(-1, 1).to_sparse()

        elif self.norm_type == "sym_block_norm":
            # D computed based on the block diagonal
            D = A.to_dense().view((num_nodes, d, num_nodes, d))
            D = torch.permute(D, (0, 2, 1, 3))  # num_nodes x num_nodes x d x d
            D = torch.diagonal(D, dim1=0, dim2=1)  # d x d x num_nodes
            D = torch.permute(D, (2, 0, 1))  # num_nodes x d x d

            # compute D^-1
            if self.sheaf_type in ["GeneralSheafs", "LowRankSheafs"]:
                D = utils.batched_sym_matrix_pow(D, -0.5)  # num_nodes x d x d
            else:
                D = torch.pow(D, -0.5)
                D[D == float("inf")] = 0
            D = torch.block_diag(*torch.unbind(D, 0))
            D = D.to_sparse()

            # compute D^-0.5AD^-0.5
            A = torch.sparse.mm(D, A)
            A = torch.sparse.mm(A, D)
            if self.sheaf_type in ["GeneralSheafs", "LowRankSheafs"]:
                A = A.to_dense().clamp(-1, 1).to_sparse()
        return A

    def forward(self, data: HeteroHypergraph):
        """
        an l-layer GCN
        """
        do, l, m = self.do, self.l, self.m
        H = data.x

        num_nodes = data.x.shape[0]
        num_edges = data.edge_index[1].max().item() + 1

        edge_index = data.edge_index

        if self.hyperedge_attr is None:
            if hasattr(data, "hyperedge_attr"):
                self.hyperedge_attr = data.hyperedge_attr
            else:
                self.hyperedge_attr = self.init_hyperedge_attr(
                    self.init_hedge,
                    num_edges=num_edges,
                    x=H,
                    hyperedge_index=edge_index,
                )

        H = self.lin(H)
        hyperedge_attr = self.lin(self.hyperedge_attr)

        H = H.view((H.shape[0] * self.d, self.MLP_hidden))  # (N * d) x num_features
        hyperedge_attr = hyperedge_attr.view(
            (hyperedge_attr.shape[0] * self.d, self.MLP_hidden)
        )

        for i, hidden in enumerate(self.layers):
            if i == 0 or self.dynamic_sheaf:
                # compute the sheaf
                sheaf = self.sheaf_builder[i](
                    H, hyperedge_attr, edge_index, data.node_types, data.hyperedge_types
                )  # N x E x d x d

                # build the laplacian based on edges amax(F_v<e(x_v)) ~ amin(F_v<e(x_v))
                # with nondiagonal terms -F_v<e(x_v)^T F_w<e(x_w)
                # and diagonal terms \sum_e F_v<e(x_v)^T F_v<e(x_v)
                h_sheaf_index, h_sheaf_attributes = self.Laplacian(
                    H, m, self.d, edge_index, sheaf
                )

                A = torch.sparse.FloatTensor(
                    h_sheaf_index,
                    h_sheaf_attributes,
                    (num_nodes * self.d, num_nodes * self.d),
                )
                A = A.coalesce()
                A = self.normalise(A, h_sheaf_index, num_nodes, self.d)

                eye_diag = torch.ones((num_nodes * self.d))
                A = (
                        utils.sparse_diagonal(
                        eye_diag, (num_nodes * self.d, num_nodes * self.d)
                    ).to(A.device)
                        - A
                )  # I - A

            if self.left_proj:
                H = H.t().reshape(-1, self.d)
                H = self.lin_left_proj[i](H)
                H = H.reshape(-1, num_nodes * self.d).t()

            H = F.elu(hidden(A, H, m))
            if i < l - 1:
                H = F.dropout(H, do, training=self.training)

        H = H.view(self.num_nodes, -1)  # Nd x out_channels -> Nx(d*out_channels)
        if self.use_lin2:
            H = F.elu(self.lin2(H))  # Nx(d*out_channels)-> N x num_classes
        return H
