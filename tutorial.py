import torch
from torch_geometric.data import Data

from hyper_sheaf.models.sheaf_hgcn.models import SheafHyperGCN
from hyper_sheaf.models.sheaf_hgnn.models import SheafHyperGNN

if __name__ == '__main__':
    device = torch.device('cpu')

    # create a random hypergraph to run inference for
    num_nodes = 10
    num_node_types = 2
    num_hyperedge_types = 2
    features = torch.rand(num_nodes, 64)
    edge_index = torch.tensor(
        [[0, 1, 2, 0, 1, 3, 4, 1, 2, 4], [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]])
    labels = torch.randint(0, 5, (num_nodes,))
    hyperedge_types = torch.randint(0, num_hyperedge_types, (3,))
    node_types = torch.randint(0, num_node_types, (num_nodes,))
    data = Data(
        x=features,
        edge_index=edge_index,
        y=labels,
        node_types=node_types,
        hyperedge_types=hyperedge_types,
        num_hyperedges=3,
        n_x=num_nodes
    ).to(device)

    model = SheafHyperGNN(
        in_channels=64,
        out_channels=5,
        use_lin2=True,
        he_feature_builder='var1',
        sheaf_learner='type_concat',
        num_node_types=num_node_types,
        num_hyperedge_types=num_hyperedge_types
    ).to(device)

    out = model(data)
    print(out.shape)

    model = SheafHyperGCN(
        V=data.num_nodes,
        in_channels=64,
        out_channels=5,
        use_lin2=True,
        he_feat_type='var1',
        sheaf_pred_block='type_concat',
        num_node_types=num_node_types,
        num_hyperedge_types=num_hyperedge_types
    ).to(device)

    out = model(data)
    print(out.shape)
