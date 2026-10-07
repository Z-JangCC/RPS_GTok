import pytest
import torch

from gptok2.data.schema import record_from_dict, record_to_dict
from rps_gtok_consumption.data import TokenExample
from rps_gtok_consumption.protocol_audit import require_equal_parameters, scenario_observability, validate_heldout_splits
from rps_gtok_consumption.exact_backend import exact_certificate
from gptok2.canonical import canonicalize


def example(graph_id, y, tokens=None):
    return TokenExample(graph_id, "audit", "", "edge_list", tokens or ["E(0,1)"], y)


@pytest.mark.parametrize("directed", [False, True])
def test_unsorted_attributed_coo_roundtrip(directed):
    row = {"graph_id": "rows", "num_nodes": 3, "edges": [[2, 0], [1, 2], [0, 0], [2, 0]], "edge_type": [4, 5, 6, 7], "edge_attr": [[.1], [.2], [.3], [.4]], "directed": directed}
    restored = record_to_dict(record_from_dict(row))
    assert restored["edges"] == row["edges"]
    assert restored["edge_type"] == row["edge_type"]
    assert torch.allclose(torch.tensor(restored["edge_attr"]), torch.tensor(row["edge_attr"]))


def test_validate_splits_rejects_missing_validation_class():
    splits = {"train": [example("a", 0), example("b", 1)], "val": [example("c", 0)], "test": [example("d", 0), example("e", 1)]}
    with pytest.raises(ValueError, match="val.*missing"):
        validate_heldout_splits(splits, "classification")


def test_validate_splits_rejects_empty_heldout():
    with pytest.raises(ValueError, match="val"):
        validate_heldout_splits({"train": [example("a", 0)], "val": [], "test": [example("b", 0)]}, "regression")


def test_validate_splits_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        validate_heldout_splits({"train": [example("same", 1.)], "val": [example("same", 2.)], "test": [example("b", 0.)]}, "regression")


def test_scenario_observability_requires_input_changes():
    mapping = {"a": "case9", "b": "case9"}
    rows = [example("a", .98), example("b", 1.02)]
    assert scenario_observability(rows, mapping)["unobservable_cases"] == ["case9"]
    rows[1].tokens.append("LOAD=2")
    assert scenario_observability(rows, mapping)["passed"]


def test_parameter_guard():
    require_equal_parameters([{"parameter_count": 10}, {"parameter_count": 10}])
    with pytest.raises(ValueError, match="differ"):
        require_equal_parameters([{"parameter_count": 10}, {"parameter_count": 11}])


def test_exact_backend_preserves_direction_and_permutation():
    from gptok2.data.schema import record_from_dict
    from scripts.run_topology_shift_downstream import permute_record

    row = {
        "graph_id": "directed",
        "num_nodes": 3,
        "edges": [[0, 1], [1, 2]],
        "node_type": [0, 1, 0],
        "edge_type": [4, 5],
        "directed": True,
    }
    record = record_from_dict(row)
    assert exact_certificate(record) == exact_certificate(permute_record(record, 17))
    reversed_row = dict(row, edges=[[1, 0], [1, 2]])
    assert exact_certificate(record) != exact_certificate(record_from_dict(reversed_row))


def test_nauty_canonical_order_is_permutation_invariant():
    from gptok2.data.schema import record_from_dict
    from scripts.run_topology_shift_downstream import permute_record

    row = {
        "graph_id": "canonical",
        "num_nodes": 4,
        "edges": [[0, 1], [1, 2], [2, 0], [2, 3]],
        "node_type": [0, 1, 0, 2],
        "edge_type": [4, 5, 4, 6],
        "directed": True,
    }
    record = record_from_dict(row)
    left = canonicalize(record, exact=True, exact_backend="nauty").record
    right = canonicalize(permute_record(record, 29), exact=True, exact_backend="nauty").record
    assert record_to_dict(left)["edges"] == record_to_dict(right)["edges"]
    assert record_to_dict(left)["node_type"] == record_to_dict(right)["node_type"]
    assert record_to_dict(left)["edge_type"] == record_to_dict(right)["edge_type"]


def test_exact_backend_collapses_reciprocal_undirected_coo_rows():
    from gptok2.data.schema import record_from_dict
    from rps_gtok_consumption.exact_backend import build_colored_incidence_graph, exact_certificate

    row = {"graph_id": "reciprocal", "num_nodes": 3, "edges": [[0, 1], [1, 0], [1, 2], [2, 1]], "directed": False}
    record = record_from_dict(row)
    graph, _, edge_count = build_colored_incidence_graph(record)
    assert edge_count == 2
    assert graph.number_of_vertices == 3
    assert exact_certificate(record) == exact_certificate(record_from_dict({**row, "edges": [[2, 1], [1, 2], [0, 1], [1, 0]]}))


def test_exact_canonical_order_is_stable_under_relabeling():
    from gptok2.data.schema import record_from_dict
    from rps_gtok_consumption.exact_backend import exact_canonical_order
    from scripts.run_topology_shift_downstream import permute_record

    row = {"graph_id": "order", "num_nodes": 4, "edges": [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], "node_type": [0, 1, 0, 2], "directed": False}
    record = record_from_dict(row)
    permuted = permute_record(record, 37)

    def signature(item, order):
        rank = {old: new for new, old in enumerate(order)}
        nodes = tuple(int(item.node_type[old]) for old in order)
        edges = tuple(sorted((min(rank[int(u)], rank[int(v)]), max(rank[int(u)], rank[int(v)])) for u, v in item.edge_index.t().tolist()))
        return nodes, edges

    assert signature(record, exact_canonical_order(record)) == signature(permuted, exact_canonical_order(permuted))
