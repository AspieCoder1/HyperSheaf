import torch
from torch_scatter import scatter_mean, scatter_add, scatter


def compute_hyperedge_features_var1(x, e, hyperedge_index):
    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)
    es = torch.index_select(e, dim=0, index=col)

    return xs, es


def compute_hyperedge_features_var2(x, hyperedge_index):
    row, col = hyperedge_index
    e = scatter_mean(x[row], col, dim=0)

    xs = torch.index_select(x, dim=0, index=row)
    es = torch.index_select(e, dim=0, index=col)

    return xs, es


def compute_hyperedge_features_var3(x, hyperedge_index, phi):
    row, col = hyperedge_index
    x_e = phi(x)
    # sum(φ(x_v)
    e = scatter_add(x_e[row], col, dim=0)
    return torch.index_select(x, dim=0, index=row), torch.index_select(e, dim=0,
                                                                       index=col)


def compute_hyperedge_index_cp_decomp(x, hyperedge_index, cp_W, cp_V):
    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)

    xs_ones = torch.cat(
        (xs, torch.ones(xs.shape[0], 1).to(xs.device)), dim=-1
    )  # nnz x f+1
    xs_ones_proj = torch.tanh(cp_W(xs_ones))  # nnz x r
    xs_prod = scatter(xs_ones_proj, col, dim=0, reduce="mul")  # edges x r
    e = torch.relu(cp_V(xs_prod))  # edges x f
    e = e + torch.relu(scatter_add(x[row], col, dim=0))
    es = torch.index_select(e, dim=0, index=col)

    return es, xs
