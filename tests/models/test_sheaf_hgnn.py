from typing import Literal

import pytest


from hypersheaf.data import HeteroHypergraph
from hypersheaf.models.sheaf_hgnn import SheafHyperGNN


def test_sheaf_hgnn(
    hypergraph: HeteroHypergraph,
):
    model = SheafHyperGNN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        use_lin2=True,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


@pytest.mark.parametrize("init_hedge", ["rand", "avg"])
def test_sheaf_hgnn_init_hedge(
    hypergraph: HeteroHypergraph,
    init_hedge: Literal["rand", "avg"],
):
    model = SheafHyperGNN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        use_lin2=True,
        init_hedge=init_hedge,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


@pytest.mark.parametrize("sheaf_act", ["sigmoid", "tanh", "none"])
def test_sheaf_hgnn_sheaf_act(
    hypergraph: HeteroHypergraph,
    sheaf_act: Literal["sigmoid", "tanh", "none"],
):
    model = SheafHyperGNN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        use_lin2=True,
        sheaf_act=sheaf_act,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


@pytest.mark.parametrize(
    "sheaf_type", ["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"]
)
def test_sheaf_hgnn_sheaf_type(
    hypergraph: HeteroHypergraph,
    sheaf_type: Literal["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"],
):
    model = SheafHyperGNN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        use_lin2=True,
        sheaf_type=sheaf_type,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


@pytest.mark.parametrize("sheaf_learner", ["Sheaf-NSD", "Sheaf-TE", "Sheaf-ensemble"])
def test_sheaf_hgnn_sheaf_learner(
    hypergraph: HeteroHypergraph,
    sheaf_learner: Literal["Sheaf-NSD", "Sheaf-TE", "Sheaf-ensemble"],
):
    model = SheafHyperGNN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        use_lin2=True,
        sheaf_learner=sheaf_learner,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


@pytest.mark.parametrize(
    "sheaf_normtype", ["degree_norm", "block_norm", "sym_degree_norm", "sym_block_norm"]
)
def test_sheaf_hgnn_sheaf_normtype(
    hypergraph: HeteroHypergraph,
    sheaf_normtype: Literal[
        "degree_norm", "block_norm", "sym_degree_norm", "sym_block_norm"
    ],
):
    model = SheafHyperGNN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        use_lin2=True,
        sheaf_normtype=sheaf_normtype,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


@pytest.mark.parametrize("he_feat_type", ["var1", "var2", "var3", "cp_decomp"])
def test_sheaf_hgnn_he_feat_type(
    hypergraph: HeteroHypergraph,
    he_feat_type: Literal["var1", "var2", "var3", "cp_decomp"],
):
    model = SheafHyperGNN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        use_lin2=True,
        he_feat_type=he_feat_type,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


@pytest.mark.parametrize(
    "sheaf_type", ["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"]
)
def test_sheaf_hgnn_dynamic_sheaf(
    hypergraph: HeteroHypergraph,
    sheaf_type: Literal["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"],
):
    model = SheafHyperGNN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        use_lin2=True,
        dynamic_sheaf=True,
        sheaf_type=sheaf_type,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


def test_invalid_hyperedge_attr_type(hypergraph: HeteroHypergraph):
    with pytest.raises(
        ValueError,
        match="Invalid hyperedge attribute initialization type. Must be 'rand' or 'avg'.",
    ):
        SheafHyperGNN(
            in_channels=16,
            out_channels=32,
            num_node_types=hypergraph.num_node_types,
            num_hyperedge_types=hypergraph.num_hyperedge_types,
            use_lin2=True,
            init_hedge="invalid",
        )


def test_use_lin2_disabled(hypergraph: HeteroHypergraph):
    model = SheafHyperGNN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        use_lin2=False,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, hypergraph.num_node_types * model.d * 32)


@pytest.mark.parametrize(
    "sheaf_type", ["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"]
)
def test_left_proj_enabled(
    hypergraph: HeteroHypergraph,
    sheaf_type: Literal["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"],
):
    model = SheafHyperGNN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        sheaf_type=sheaf_type,
        use_lin2=True,
        left_proj=True,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


@pytest.mark.parametrize(
    "sheaf_type", ["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"]
)
def test_bias_disabled(
    hypergraph: HeteroHypergraph,
    sheaf_type: Literal["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"],
):
    model = SheafHyperGNN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        sheaf_type=sheaf_type,
        use_lin2=True,
        bias=False,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


@pytest.mark.parametrize(
    "sheaf_type", ["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"]
)
def test_residual_enabled(
    hypergraph: HeteroHypergraph,
    sheaf_type: Literal["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"],
):
    model = SheafHyperGNN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        sheaf_type=sheaf_type,
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
    model = SheafHyperGNN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        use_lin2=True,
        sheaf_type=sheaf_type,
        sheaf_normtype=sheaf_normtype,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


@pytest.mark.parametrize(
    "sheaf_type", ["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"]
)
def test_sheaf_special_head(
    hypergraph: HeteroHypergraph,
    sheaf_type: Literal["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"],
):
    model = SheafHyperGNN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        use_lin2=True,
        sheaf_type=sheaf_type,
        sheaf_special_head=True,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)


@pytest.mark.parametrize(
    "sheaf_type", ["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"]
)
def test_sheaf_special_dropout(
    hypergraph: HeteroHypergraph,
    sheaf_type: Literal["DiagSheafs", "OrthoSheafs", "GeneralSheafs", "LowRankSheafs"],
):
    model = SheafHyperGNN(
        in_channels=16,
        out_channels=32,
        num_node_types=hypergraph.num_node_types,
        num_hyperedge_types=hypergraph.num_hyperedge_types,
        use_lin2=True,
        sheaf_type=sheaf_type,
        sheaf_dropout=True,
    )

    out = model(hypergraph)
    assert out.shape == (hypergraph.num_nodes, 32)
