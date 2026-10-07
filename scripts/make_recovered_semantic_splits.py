"""Make deterministic, label-stratified splits from audited recovered records."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from gptok2.data.io import load_records
from gptok2_tokenizer import GPTok2Tokenizer


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--datasets", nargs="+", default=["cifar10_sp", "mnist_sp", "mutag", "ogbg_molhiv"]); p.add_argument("--out", type=Path, default=ROOT / "runs/recovered_semantic_splits")
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    report = []
    for dataset in args.datasets:
        path = ROOT / f"rebuttal/data_21_labeled/{dataset}.jsonl"
        records = load_records(path)
        groups = defaultdict(list)
        for row in records:
            if row.y is None or row.y.numel() != 1:
                raise ValueError(f"{dataset}: non-scalar or missing y")
            groups[int(row.y.reshape(-1)[0].item())].append(row)
        rng = random.Random(20261005)
        for rows in groups.values(): rng.shuffle(rows)
        splits = {"train": [], "val": [], "test": []}
        for rows in groups.values():
            n = len(rows); n_test = max(1, round(n * 0.2)); n_val = max(1, round(n * 0.2))
            if n - n_test - n_val < 1:
                n_val = max(0, n - n_test - 1)
            splits["test"].extend(rows[:n_test]); splits["val"].extend(rows[n_test:n_test+n_val]); splits["train"].extend(rows[n_test+n_val:])
        for split, rows in splits.items():
            rng.shuffle(rows)
            out = args.out / dataset / f"{split}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(json.dumps({"graph_id": r.graph_id, "num_nodes": int(r.num_nodes), "edges": r.edge_index.t().tolist() if r.edge_index.numel() else [], "node_type": r.node_type.tolist() if r.node_type is not None else None, "edge_type": r.edge_type.tolist() if r.edge_type is not None else None, "node_attr": r.node_attr.tolist() if r.node_attr is not None else None, "edge_attr": r.edge_attr.tolist() if r.edge_attr is not None else None, "y": r.y.tolist() if r.y is not None else None, "metadata": r.metadata, "directed": r.directed}, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
        tokenizer = GPTok2Tokenizer({"canonicalization": {"enabled": False, "max_search_nodes": 20, "timeout_sec": 0.2}, "program": {"noncanonical_order": True, "max_global_links": 32}, "patch": {"max_patches_per_graph": 32, "grow_rounds": 1, "max_size": 5, "max_ports": 4, "max_cycle_signatures": 8, "max_triangle_signatures": 16}, "vq": {"num_codes": 128}}).fit(splits["train"])
        tokenizer.save(args.out / dataset / "tokenizer.json")
        all_classes = sorted(groups)
        split_classes = {k: sorted({int(r.y.item()) for r in v}) for k, v in splits.items()}
        report.append({"dataset": dataset, "counts": {k: len(v) for k, v in splits.items()}, "classes": all_classes, "split_classes": split_classes, "eligible_for_classification": all(set(split_classes[k]) == set(all_classes) for k in ("train", "val", "test"))})
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
