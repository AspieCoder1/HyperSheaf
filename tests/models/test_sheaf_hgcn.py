import pytest
from typing import Literal

from hypersheaf.data import HeteroHypergraph
from hypersheaf.models.sheaf_hgcn.models import SheafHyperGCN


def test_sheaf_hgcn(hypergraph: HeteroHypergraph):
    model = SheafHyperGCN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        num_nodes=hypergraph.num_nodes,
        use_lin2=True,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


@pytest.mark.parametrize(
    "sheaf_type", ["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"]
)
def test_sheaf_hgcn_sheaf_type(hypergraph: HeteroHypergraph, sheaf_type: str):
    model = SheafHyperGCN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        num_nodes=hypergraph.num_nodes,
        use_lin2=True,
        sheaf_type=sheaf_type,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


@pytest.mark.parametrize(
    "sheaf_type", ["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"]
)
def test_sheaf_hgcn_left_proj(hypergraph: HeteroHypergraph, sheaf_type: str):
    model = SheafHyperGCN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        num_nodes=hypergraph.num_nodes,
        use_lin2=True,
        sheaf_type=sheaf_type,
        left_proj=True,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


@pytest.mark.parametrize(
    "sheaf_type", ["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"]
)
def test_sheaf_hgcn_dynamic_sheaf(hypergraph: HeteroHypergraph, sheaf_type: str):
    model = SheafHyperGCN(
        in_channels=16,
        out_channels=32,
        hidden_channels=16,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        num_nodes=hypergraph.num_nodes,
        use_lin2=True,
        sheaf_type=sheaf_type,
        dynamic_sheaf=True,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


@pytest.mark.parametrize(
    "sheaf_type", ["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"]
)
def test_residual_enabled(hypergraph: HeteroHypergraph, sheaf_type: str):
    model = SheafHyperGCN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        num_nodes=hypergraph.num_nodes,
        use_lin2=True,
        residual=True,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


@pytest.mark.parametrize(
    "sheaf_type", ["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"]
)
@pytest.mark.parametrize(
    "sheaf_normtype", ["degree_norm", "block_norm", "sym_degree_norm", "sym_block_norm"]
)
def test_sheaf_type_sheaf_norm_type_combinations(
    hypergraph: HeteroHypergraph,
    sheaf_normtype: Literal[
        "degree_norm", "block_norm", "sym_degree_norm", "sym_block_norm"
    ],
    sheaf_type: Literal["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"],
):
    model = SheafHyperGCN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        use_lin2=True,
        sheaf_type=sheaf_type,
        sheaf_normtype=sheaf_normtype,
        num_nodes=hypergraph.num_nodes,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


@pytest.mark.parametrize("he_feat_type", ["var1", "var2", "var3", "cp_decomp"])
def test_sheaf_hgnn_he_feat_type(
    hypergraph: HeteroHypergraph,
    he_feat_type: Literal["var1", "var2", "var3", "cp_decomp"],
):
    model = SheafHyperGCN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        use_lin2=True,
        he_feat_type=he_feat_type,
        num_nodes=hypergraph.num_nodes,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)
