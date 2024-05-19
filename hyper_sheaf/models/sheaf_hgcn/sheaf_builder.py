import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_add, scatter

from hyper_sheaf.models.mlp import MLP
from hyper_sheaf.utils.orthogonal import Orthogonal


# helper functions to predict sigma(MLP(x_v || h_e)) varying how thw attributes for hyperedge are computed
def predict_blocks(x, e, hyperedge_index, sheaf_lin, args):
    # e_j = avg(x_v)
    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)
    es = torch.index_select(e, dim=0, index=col)

    # sigma(MLP(x_v || h_e))
    h_sheaf = torch.cat((xs, es), dim=-1)  # sparse version of an NxEx2f tensor
    h_sheaf = sheaf_lin(h_sheaf)  # sparse version of an NxExd tensor
    if args.sheaf_act == 'sigmoid':
        h_sheaf = F.sigmoid(
            h_sheaf)  # output d numbers for every entry in the incidence matrix
    elif args.sheaf_act == 'tanh':
        h_sheaf = F.tanh(
            h_sheaf)  # output d numbers for every entry in the incidence matrix
    return h_sheaf


def predict_blocks_var2(x, hyperedge_index, sheaf_lin, args):
    # e_j = avg(h_v)
    row, col = hyperedge_index
    e = scatter_mean(x[row], col, dim=0)

    xs = torch.index_select(x, dim=0, index=row)
    es = torch.index_select(e, dim=0, index=col)

    # sigma(MLP(x_v || h_e))
    h_sheaf = torch.cat((xs, es), dim=-1)  # sparse version of an NxEx2f tensor
    h_sheaf = sheaf_lin(h_sheaf)  # sparse version of an NxExd tensor
    if args.sheaf_act == 'sigmoid':
        h_sheaf = F.sigmoid(
            h_sheaf)  # output d numbers for every entry in the incidence matrix
    elif args.sheaf_act == 'tanh':
        h_sheaf = F.tanh(
            h_sheaf)  # output d numbers for every entry in the incidence matrix

    return h_sheaf


def predict_blocks_var3(x, hyperedge_index, sheaf_lin, sheaf_lin2, args):
    # universal approx according to  Equivariant Hypergraph Diffusion Neural Operators
    # # e_j = sum(φ(x_v))

    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)

    # φ(x_v)
    x_e = sheaf_lin2(x)
    # sum(φ(x_v)
    e = scatter_add(x_e[row], col, dim=0)
    es = torch.index_select(e, dim=0, index=col)

    # sigma(MLP(x_v || h_e))
    h_sheaf = torch.cat((xs, es), dim=-1)  # sparse v ersion of an NxEx2f tensor
    h_sheaf = sheaf_lin(h_sheaf)  # sparse version of an NxExd tensor
    if args.sheaf_act == 'sigmoid':
        h_sheaf = F.sigmoid(
            h_sheaf)  # output d numbers for every entry in the incidence matrix
    elif args.sheaf_act == 'tanh':
        h_sheaf = F.tanh(
            h_sheaf)  # output d numbers for every entry in the incidence matrix

    return h_sheaf


def predict_blocks_cp_decomp(x, hyperedge_index, cp_W, cp_V, sheaf_lin, args):
    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)

    xs_ones = torch.cat((xs, torch.ones(xs.shape[0], 1).to(xs.device)),
                        dim=-1)  # nnz x f+1
    xs_ones_proj = torch.tanh(cp_W(xs_ones))  # nnz x r
    xs_prod = scatter(xs_ones_proj, col, dim=0, reduce="mul")  # edges x r
    e = torch.relu(cp_V(xs_prod))  # edges x f
    e = e + torch.relu(scatter_add(x[row], col, dim=0))
    es = torch.index_select(e, dim=0, index=col)

    # sigma(MLP(x_v || h_e))
    h_sheaf = torch.cat((xs, es), dim=-1)  # sparse version of an NxEx2f tensor
    h_sheaf = sheaf_lin(h_sheaf)  # sparse version of an NxExd tensor
    if args.sheaf_act == 'sigmoid':
        h_sheaf = F.sigmoid(
            h_sheaf)  # output d numbers for every entry in the incidence matrix
    elif args.sheaf_act == 'tanh':
        h_sheaf = F.tanh(
            h_sheaf)  # output d numbers for every entry in the incidence matrix
    return h_sheaf


# the hidden dimensiuon are a bit differently computed
# That's why the classes are separated.
# We shoul merge them at some point
# Moreover, the function only return the values, not the indices in this case


class HGCNSheafBuilderDiag(nn.Module):
    def __init__(
            self,
            stalk_dimension: int,
            hidden_channels: int = 64,
            dropout: float = 0.6,
            allset_input_norm: bool = True,
            sheaf_special_head: bool = False,
            sheaf_pred_block: str = "MLP_var1",
            sheaf_dropout: bool = False,
            sheaf_act: str = "sigmoid",
            **_kwargs,
    ):
        """
        hidden_dim overwrite the self.MLP_hidden used in the normal sheaf HNN
        """
        super(HGCNSheafBuilderDiag, self).__init__()
        self.prediction_type = (
            sheaf_pred_block  # pick the way hyperedge feartures are computed
        )
        self.sheaf_dropout = sheaf_dropout
        self.special_head = sheaf_special_head  # add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
        self.d = stalk_dimension  # stalk dimension
        self.MLP_hidden = hidden_channels
        self.norm = allset_input_norm
        self.dropout = dropout
        self.sheaf_act = sheaf_act

        if self.prediction_type == "MLP_var1":
            self.sheaf_lin = MLP(
                in_channels=2 * self.MLP_hidden,
                hidden_channels=self.MLP_hidden,
                out_channels=self.d,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
        else:
            self.sheaf_lin = MLP(
                in_channels=2 * self.MLP_hidden,
                hidden_channels=self.MLP_hidden,
                out_channels=self.d,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
        if self.prediction_type == "MLP_var3":
            self.sheaf_lin2 = MLP(
                in_channels=self.MLP_hidden,
                hidden_channels=self.MLP_hidden,
                out_channels=self.MLP_hidden,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
        if self.prediction_type == "cp_decomp":
            self.cp_W = MLP(
                in_channels=self.MLP_hidden + 1,
                hidden_channels=self.MLP_hidden,
                out_channels=self.MLP_hidden,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
            self.cp_V = MLP(
                in_channels=self.MLP_hidden,
                hidden_channels=self.MLP_hidden,
                out_channels=self.MLP_hidden,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.MLP_hidden,
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
    def forward(self, x, e, hyperedge_index):
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

        if self.prediction_type == "MLP_var1":
            h_sheaf = predict_blocks(
                x, e, hyperedge_index, self.sheaf_lin, self.sheaf_act
            )
        elif self.prediction_type == "MLP_var2":
            h_sheaf = predict_blocks_var2(
                x, hyperedge_index, self.sheaf_lin, self.sheaf_act
            )
        elif self.prediction_type == "MLP_var3":
            h_sheaf = predict_blocks_var3(
                x, hyperedge_index, self.sheaf_lin, self.sheaf_lin2, self.sheaf_act
            )
        elif self.prediction_type == "cp_decomp":
            h_sheaf = predict_blocks_cp_decomp(
                x, hyperedge_index, self.cp_W, self.cp_V, self.sheaf_lin, self.sheaf_act
            )

        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)

        return h_sheaf


class HGCNSheafBuilderGeneral(nn.Module):
    def __init__(
            self,
            stalk_dimension: int,
            hidden_channels: int = 64,
            dropout: float = 0.6,
            allset_input_norm: bool = True,
            sheaf_special_head: bool = False,
            sheaf_pred_block: str = "MLP_var1",
            sheaf_dropout: bool = False,
            sheaf_act: str = "sigmoid",
            **_kwargs,
    ):
        super(HGCNSheafBuilderGeneral, self).__init__()
        self.prediction_type = (
            sheaf_pred_block  # pick the way hyperedge feartures are computed
        )
        self.sheaf_dropout = sheaf_dropout
        self.special_head = sheaf_special_head  # add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
        self.d = stalk_dimension  # stalk dimension
        self.MLP_hidden = hidden_channels
        self.norm = allset_input_norm
        self.dropout = dropout
        self.sheaf_act = sheaf_act

        if self.prediction_type == "MLP_var1":
            self.sheaf_lin = MLP(
                in_channels=self.MLP_hidden + self.MLP_hidden,
                hidden_channels=self.MLP_hidden,
                out_channels=self.d * self.d,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
        else:
            self.sheaf_lin = MLP(
                in_channels=2 * self.MLP_hidden,
                hidden_channels=self.MLP_hidden,
                out_channels=self.d * self.d,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
        if self.prediction_type == "MLP_var3":
            self.sheaf_lin2 = MLP(
                in_channels=self.MLP_hidden,
                hidden_channels=self.MLP_hidden,
                out_channels=self.MLP_hidden,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
        if self.prediction_type == "cp_decomp":
            self.cp_W = MLP(
                in_channels=self.MLP_hidden + 1,
                hidden_channels=self.MLP_hidden,
                out_channels=self.MLP_hidden,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
            self.cp_V = MLP(
                in_channels=self.MLP_hidden,
                hidden_channels=self.MLP_hidden,
                out_channels=self.MLP_hidden,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.MLP_hidden,
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
    def forward(self, x, e, hyperedge_index):
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

        # h_sheaf = self.predict_blocks(x, e, hyperedge_index, sheaf_lin)
        # h_sheaf = self.predict_blocks_var2(x, hyperedge_index, sheaf_lin)
        if self.prediction_type == "MLP_var1":
            h_sheaf = predict_blocks(
                x, e, hyperedge_index, self.sheaf_lin, self.sheaf_act
            )
        elif self.prediction_type == "MLP_var2":
            h_sheaf = predict_blocks_var2(
                x, hyperedge_index, self.sheaf_lin, self.sheaf_act
            )
        elif self.prediction_type == "MLP_var3":
            h_sheaf = predict_blocks_var3(
                x, hyperedge_index, self.sheaf_lin, self.sheaf_lin2, self.sheaf_act
            )
        elif self.prediction_type == "cp_decomp":
            h_sheaf = predict_blocks_cp_decomp(
                x, hyperedge_index, self.cp_W, self.cp_V, self.sheaf_lin, self.sheaf_act
            )

        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)

        return h_sheaf


class HGCNSheafBuilderOrtho(nn.Module):
    def __init__(
            self,
            stalk_dimension: int,
            hidden_channels: int = 64,
            dropout: float = 0.6,
            allset_input_norm: bool = True,
            sheaf_special_head: bool = False,
            sheaf_pred_block: str = "MLP_var1",
            sheaf_dropout: bool = False,
            sheaf_act: str = "sigmoid",
            **_kwargs,
    ):
        super(HGCNSheafBuilderOrtho, self).__init__()
        self.prediction_type = (
            sheaf_pred_block  # pick the way hyperedge feartures are computed
        )
        self.sheaf_dropout = sheaf_dropout
        self.special_head = sheaf_special_head  # add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
        self.d = stalk_dimension
        self.MLP_hidden = hidden_channels
        self.norm = allset_input_norm
        self.sheaf_act = sheaf_act

        self.orth_transform = Orthogonal(
            d=self.d, orthogonal_map="householder"
        )  # method applied to transform params into ortho dxd matrix
        self.dropout = dropout

        if self.prediction_type == "MLP_var1":
            self.sheaf_lin = MLP(
                in_channels=self.MLP_hidden + self.MLP_hidden,
                hidden_channels=self.MLP_hidden,
                out_channels=self.d * (self.d - 1) // 2,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
        else:
            self.sheaf_lin = MLP(
                in_channels=2 * self.MLP_hidden,
                hidden_channels=self.MLP_hidden,
                out_channels=self.d * (self.d - 1) // 2,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
        if self.prediction_type == "MLP_var3":
            self.sheaf_lin2 = MLP(
                in_channels=self.MLP_hidden,
                hidden_channels=self.MLP_hidden,
                out_channels=self.MLP_hidden,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
        if self.prediction_type == "cp_decomp":
            self.cp_W = MLP(
                in_channels=self.MLP_hidden + 1,
                hidden_channels=self.MLP_hidden,
                out_channels=self.MLP_hidden,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
            self.cp_V = MLP(
                in_channels=self.MLP_hidden,
                hidden_channels=self.MLP_hidden,
                out_channels=self.MLP_hidden,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.MLP_hidden,
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
    def forward(self, x, e, hyperedge_index):
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

        if self.prediction_type == "MLP_var1":
            h_sheaf = predict_blocks(
                x, e, hyperedge_index, self.sheaf_lin, self.sheaf_act
            )
        elif self.prediction_type == "MLP_var2":
            h_sheaf = predict_blocks_var2(
                x, hyperedge_index, self.sheaf_lin, self.sheaf_act
            )
        elif self.prediction_type == "MLP_var3":
            h_sheaf = predict_blocks_var3(
                x, hyperedge_index, self.sheaf_lin, self.sheaf_lin2, self.sheaf_act
            )
        elif self.prediction_type == "cp_decomp":
            h_sheaf = predict_blocks_cp_decomp(
                x, hyperedge_index, self.cp_W, self.cp_V, self.sheaf_lin, self.sheaf_act
            )

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
            sheaf_pred_block: str = "MLP_var1",
            sheaf_dropout: bool = False,
            sheaf_act: str = "sigmoid",
            rank: int = 2,
            **_kwargs,
    ):
        super(HGCNSheafBuilderLowRank, self).__init__()
        self.prediction_type = (
            sheaf_pred_block  # pick the way hyperedge feartures are computed
        )
        self.sheaf_dropout = sheaf_dropout
        self.special_head = sheaf_special_head  # add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
        self.d = stalk_dimension  # stalk dimension
        self.MLP_hidden = hidden_channels
        self.norm = allset_input_norm
        self.sheaf_act = sheaf_act

        self.rank = rank
        self.dropout = dropout

        if self.prediction_type == "MLP_var1":
            self.sheaf_lin = MLP(
                in_channels=self.MLP_hidden + self.MLP_hidden,
                hidden_channels=self.MLP_hidden,
                out_channels=2 * self.d * self.rank + self.d,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
        else:
            self.sheaf_lin = MLP(
                in_channels=2 * self.MLP_hidden,
                hidden_channels=self.MLP_hidden,
                out_channels=2 * self.d * self.rank + self.d,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
        if self.prediction_type == "MLP_var3":
            self.sheaf_lin2 = MLP(
                in_channels=self.MLP_hidden,
                hidden_channels=self.MLP_hidden,
                out_channels=self.MLP_hidden,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
        if self.prediction_type == "cp_decomp":
            self.cp_W = MLP(
                in_channels=self.MLP_hidden + 1,
                hidden_channels=self.MLP_hidden,
                out_channels=self.MLP_hidden,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.norm,
            )
            self.cp_V = MLP(
                in_channels=self.MLP_hidden,
                hidden_channels=self.MLP_hidden,
                out_channels=self.MLP_hidden,
                num_layers=1,
                dropout=0.0,
                normalisation="ln",
                input_norm=self.MLP_hidden,
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
    def forward(self, x, e, hyperedge_index):
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

        # h_sheaf = self.predict_blocks(x, e, hyperedge_index, sheaf_lin)
        # h_sheaf = self.predict_blocks_var2(x, hyperedge_index, sheaf_lin)
        if self.prediction_type == "MLP_var1":
            h_sheaf = predict_blocks(
                x, e, hyperedge_index, self.sheaf_lin, self.sheaf_act
            )
        elif self.prediction_type == "MLP_var2":
            h_sheaf = predict_blocks_var2(
                x, hyperedge_index, self.sheaf_lin, self.sheaf_act
            )
        elif self.prediction_type == "MLP_var3":
            h_sheaf = predict_blocks_var3(
                x, hyperedge_index, self.sheaf_lin, self.sheaf_lin2, self.sheaf_act
            )
        elif self.prediction_type == "cp_decomp":
            h_sheaf = predict_blocks_cp_decomp(
                x, hyperedge_index, self.cp_W, self.cp_V, self.sheaf_lin, self.sheaf_act
            )

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
