import torch
from torch.nn import functional as F


def predict_block_local_concat(xs, es, sheaf_lin, sheaf_act):
    # sigma(MLP(x_v || h_e || t_v || t_u)))
    h_sheaf = torch.cat((xs, es), dim=-1)
    h_sheaf = sheaf_lin(h_sheaf)
    if sheaf_act == "sigmoid":
        h_sheaf = F.sigmoid(
            h_sheaf
        )  # output d numbers for every entry in the incidence matrix
    elif sheaf_act == "tanh":
        h_sheaf = F.tanh(
            h_sheaf
        )  # output d numbers for every entry in the incidence matrix
    elif sheaf_act == "elu":
        h_sheaf = F.elu(h_sheaf)
    return h_sheaf


def predict_block_type_concat(xs, es, hyperedge_index, node_types, hyperedge_types,
                              sheaf_lin, sheaf_act):
    node, hyperedge = hyperedge_index
    node_types_onehot = F.one_hot(node_types.to(torch.long))
    hyperedge_types_onehot = F.one_hot(hyperedge_types.to(torch.long))
    x_type = torch.index_select(node_types_onehot, dim=0, index=node)
    e_type = torch.index_select(hyperedge_types_onehot, dim=0, index=hyperedge)

    # sigma(MLP(x_v || h_e || t_v || t_u)))
    h_sheaf = torch.cat((xs, es, x_type, e_type), dim=-1)
    h_sheaf = sheaf_lin(h_sheaf)
    if sheaf_act == "sigmoid":
        h_sheaf = F.sigmoid(
            h_sheaf
        )  # output d numbers for every entry in the incidence matrix
    elif sheaf_act == "tanh":
        h_sheaf = F.tanh(
            h_sheaf
        )  # output d numbers for every entry in the incidence matrix
    elif sheaf_act == "elu":
        h_sheaf = F.elu(h_sheaf)

    print(h_sheaf.shape)
    return h_sheaf


def predict_block_type_ensemble(xs, es, hyperedge_index, hyperedge_types, sheaf_lins,
                                sheaf_act):
    node, hyperedge = hyperedge_index
    h_cat = torch.cat((xs, es), dim=-1)

    hyperedge_types = torch.index_select(hyperedge_types, dim=0, index=hyperedge).to(
        torch.long)

    unique, counts = torch.unique(hyperedge_types, return_counts=True)
    hyperedge_type_idx = torch.argsort(hyperedge_types)
    hyperedge_type_splits = hyperedge_type_idx.split(split_size=counts.tolist())

    results = []

    for i, split in enumerate(hyperedge_type_splits):
        results.append(sheaf_lins[i](h_cat[split]))

    stacked_maps = torch.row_stack(results)
    h_sheaf = torch.empty(stacked_maps.shape, device=stacked_maps.device)
    h_sheaf[hyperedge_type_idx] = stacked_maps

    if sheaf_act == "sigmoid":
        h_sheaf = F.sigmoid(
            h_sheaf
        )  # output d numbers for every entry in the incidence matrix
    elif sheaf_act == "tanh":
        h_sheaf = F.tanh(
            h_sheaf
        )  # output d numbers for every entry in the incidence matrix
    elif sheaf_act == "elu":
        h_sheaf = F.elu(h_sheaf)
    return h_sheaf
