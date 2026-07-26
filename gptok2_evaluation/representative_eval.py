from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import networkx as nx
import torch

from gptok2.data.io import save_records
from gptok2.data.schema import GraphRecord, edge_set, graph_to_record, to_networkx
from gptok2.metrics.evaluate import structure_fidelity
from gptok2.patches.proposal import patches_for_records
from gptok2.program.actions import GraphProgram, parse_token
from gptok2.program.compiler import compile_program
from gptok2.program.interpreter import execute_program
from gptok2.utils.io import ensure_dir
from gptok2.vq.codebook import learn_codebook
from gptok2_compact import CompactCodec, EntropyModel
from gptok2_compact.codec import compact_symbol_bits, original_program_bits
from gptok2_motif_macro import MotifEntropyModel
from gptok2_motif_macro import MotifMacroCodec
from gptok2_motif_macro import motif_symbol_bits
from gptok2_tokenizer import GPTok2Tokenizer


MODES = ["original", "compact", "entropy", "motif_macro", "motif_entropy", "motif_hybrid"]
DATASETS = ["MUTAG", "PROTEINS", "IMDB-BINARY", "QM9", "ZINC", "OGBG-MOLHIV", "PEPTIDES-FUNC"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the final tokenizer on representative graph datasets.")
    parser.add_argument("--out", default="runs/rps_gtok_representative_eval")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train", type=int, default=300)
    parser.add_argument("--max-val", type=int, default=80)
    parser.add_argument("--max-test", type=int, default=120)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--cora-ego-samples", type=int, default=500)
    parser.add_argument("--cora-ego-radius", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = ensure_dir(args.out)
    cfg = vars(args)
    (root / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    all_rows: list[dict[str, Any]] = []
    dataset_status: list[dict[str, Any]] = []
    for dataset in [normalize_dataset_name(d) for d in args.datasets]:
        out_dir = ensure_dir(root / dataset)
        summary_path = out_dir / "summary.csv"
        if summary_path.exists() and not args.overwrite:
            print(f"[skip] {dataset}: summary exists", flush=True)
            all_rows.extend(read_csv(summary_path))
            status_path = out_dir / "dataset_status.json"
            if status_path.exists():
                dataset_status.append(json.loads(status_path.read_text(encoding="utf-8")))
            continue
        print(f"[dataset] {dataset}: loading", flush=True)
        try:
            splits, status = load_dataset_splits(dataset, out_dir, args)
            status["run_status"] = "loaded"
            status["dataset"] = dataset
            status["train_graphs"] = len(splits["train"])
            status["val_graphs"] = len(splits["val"])
            status["test_graphs"] = len(splits["test"])
            (out_dir / "dataset_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            rows = run_dataset(dataset, splits, out_dir)
            write_csv(summary_path, rows)
            all_rows.extend(rows)
            dataset_status.append(status)
        except Exception as exc:
            status = {"dataset": dataset, "run_status": "failed", "reason": repr(exc)}
            (out_dir / "dataset_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            dataset_status.append(status)
            all_rows.append({"dataset": dataset, "split": "dataset", "mode": "loader", "run_status": "failed", "reason": repr(exc)})
            print(f"[failed] {dataset}: {exc!r}", flush=True)

    write_csv(root / "summary_all.csv", all_rows)
    test_rows = [r for r in all_rows if r.get("split") == "test"]
    write_csv(root / "test_comparison.csv", test_rows)
    write_csv(root / "dataset_status.csv", dataset_status)
    print(f"[done] representative evaluation: {root}", flush=True)


def normalize_dataset_name(name: str) -> str:
    n = str(name).upper().replace("_", "-")
    if n in {"OGBG-MOLHIV", "MOLHIV", "OGB-MOLHIV"}:
        return "OGBG-MOLHIV"
    if n in {"PEPTIDES-FUNC", "PEPTIDES_FUNC", "PEPTIDES"}:
        return "PEPTIDES-FUNC"
    if n == "IMDB_BINARY":
        return "IMDB-BINARY"
    return n


def prepare_real_tu_dataset(config: dict[str, Any], out_dir: Path, name: str, url: str | None = None) -> dict[str, list[GraphRecord]]:
    name = str(name).upper()
    raw_dir = ensure_dir(out_dir / "data" / "real_raw" / name)
    zip_path = raw_dir / f"{name}.zip"
    if not zip_path.exists():
        url = url or f"https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip"
        with urllib.request.urlopen(url, timeout=60) as resp:
            zip_path.write_bytes(resp.read())
    extract_dir = raw_dir / name
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
    records = _parse_tu_dataset(extract_dir, name)
    rng = random.Random(int(config.get("run", {}).get("seed", 42)))
    rng.shuffle(records)
    split_cfg = config.get("data", {}).get("split", {})
    n = len(records)
    train_n = int(round(float(split_cfg.get("train_ratio", 0.7)) * n))
    val_n = int(round(float(split_cfg.get("val_ratio", 0.1)) * n))
    splits = {"train": records[:train_n], "val": records[train_n : train_n + val_n], "test": records[train_n + val_n :]}
    for split, rows in splits.items():
        save_records(out_dir / "data" / f"real_{name.lower()}_{split}.jsonl", rows)
    return splits


def _parse_tu_dataset(folder: Path, name: str) -> list[GraphRecord]:
    indicator = _read_int_lines(folder / f"{name}_graph_indicator.txt")
    labels = _read_int_lines(folder / f"{name}_graph_labels.txt")
    edges = []
    with (folder / f"{name}_A.txt").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                a, b = [int(x.strip()) for x in line.split(",")[:2]]
                edges.append((a - 1, b - 1))
    node_to_graph = {i: gid - 1 for i, gid in enumerate(indicator)}
    by_graph_nodes: dict[int, list[int]] = {}
    by_graph_edges: dict[int, list[tuple[int, int]]] = {}
    for node, gid in node_to_graph.items():
        by_graph_nodes.setdefault(gid, []).append(node)
    for u, v in edges:
        gu, gv = node_to_graph[u], node_to_graph[v]
        if gu == gv:
            by_graph_edges.setdefault(gu, []).append((u, v))
    out: list[GraphRecord] = []
    label_values = sorted(set(labels))
    label_to_id = {v: i for i, v in enumerate(label_values)}
    for gid in sorted(by_graph_nodes):
        nodes = sorted(by_graph_nodes[gid])
        mapping = {n: i for i, n in enumerate(nodes)}
        graph = nx.Graph()
        graph.add_nodes_from(range(len(nodes)))
        graph.add_edges_from((mapping[u], mapping[v]) for u, v in by_graph_edges.get(gid, []))
        dataset_id = name.lower()
        record = graph_to_record(graph, f"{dataset_id}_{gid:04d}", dataset_id)
        record.y = torch.tensor(label_to_id[labels[gid]], dtype=torch.long)
        record.metadata.update({"family": dataset_id, "real_dataset": name, "class_label": int(record.y.item())})
        out.append(record)
    return out


def _read_int_lines(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def load_dataset_splits(dataset: str, out_dir: Path, args: argparse.Namespace) -> tuple[dict[str, list[GraphRecord]], dict[str, Any]]:
    seed = int(args.seed)
    if dataset in {"MUTAG", "PROTEINS", "IMDB-BINARY"}:
        config = {"run": {"seed": seed}, "data": {"split": {"train_ratio": 0.7, "val_ratio": 0.1}}}
        splits = prepare_real_tu_dataset(config, out_dir, dataset)
        return cap_splits(splits, args, seed), {"loader": "TU Dortmund raw parser", "source_dataset": dataset}
    if dataset == "QM9":
        processed = first_existing(
            [
                out_dir / "pyg_qm9" / "processed" / "data_v3.pt",
                out_dir / "pyg_qm9" / "processed" / "data.pt",
            ]
        )
        if processed is not None:
            records, source_size = processed_pyg_records(
                processed,
                "qm9",
                args.max_nodes,
                seed=seed,
                sample_cap=max(1, int(args.max_train) + int(args.max_val) + int(args.max_test)),
            )
            return split_records(records, args, seed), {
                "loader": "cached torch_geometric processed QM9",
                "processed_path": "<repo-local-cache>",
                "source_size": source_size,
            }
        from torch_geometric.datasets import QM9

        pyg = QM9(str(out_dir / "pyg_qm9"))
        records = pyg_to_records(pyg, "qm9", args.max_nodes)
        return split_records(records, args, seed), {"loader": "torch_geometric.datasets.QM9", "source_size": len(pyg)}
    if dataset == "ZINC":
        processed_root = first_existing(
            [
                out_dir / "pyg_zinc" / "subset" / "processed",
            ]
        )
        if processed_root is not None and all((processed_root / f"{split}.pt").exists() for split in ["train", "val", "test"]):
            splits = {}
            source_size = 0
            for split in ["train", "val", "test"]:
                records, split_size = processed_pyg_records(
                    processed_root / f"{split}.pt",
                    f"zinc_{split}",
                    args.max_nodes,
                    seed=seed,
                    sample_cap=getattr(args, f"max_{split}"),
                )
                source_size += split_size
                splits[split] = cap_list(records, getattr(args, f"max_{split}"), seed)
            return splits, {
                "loader": "cached torch_geometric processed ZINC(subset)",
                "processed_path": "<repo-local-cache>",
                "source_size": source_size,
            }
        from torch_geometric.datasets import ZINC

        records = []
        source_size = 0
        for split in ["train", "val", "test"]:
            pyg = ZINC(str(out_dir / "pyg_zinc"), subset=True, split=split)
            source_size += len(pyg)
            records.extend(pyg_to_records(pyg, f"zinc_{split}", args.max_nodes))
        return split_records(records, args, seed), {"loader": "torch_geometric.datasets.ZINC(subset=True)", "source_size": source_size}
    if dataset == "CORA":
        from torch_geometric.datasets import Planetoid

        pyg = Planetoid(str(out_dir / "pyg_cora"), name="Cora")[0]
        records = cora_ego_records(pyg, args.cora_ego_samples, args.cora_ego_radius, args.max_nodes, seed)
        return split_records(records, args, seed), {
            "loader": "torch_geometric.datasets.Planetoid(Cora) converted to ego-graph samples",
            "source_size": 1,
            "ego_samples": len(records),
            "ego_radius": args.cora_ego_radius,
        }
    if dataset == "OGBG-MOLHIV":
        from ogb.graphproppred import PygGraphPropPredDataset

        pyg = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(out_dir / "ogb"))
        split_idx = pyg.get_idx_split()
        splits = {}
        for split, key in [("train", "train"), ("val", "valid"), ("test", "test")]:
            indices = split_idx[key].tolist()
            records = pyg_to_records((pyg[i] for i in indices), f"ogbg_molhiv_{split}", args.max_nodes)
            splits[split] = cap_list(records, getattr(args, f"max_{split}"), seed)
        return splits, {"loader": "ogb.graphproppred.PygGraphPropPredDataset(ogbg-molhiv)", "source_size": len(pyg)}
    if dataset == "PEPTIDES-FUNC":
        processed = first_existing(
            [
                out_dir / "hf_geometric_data_processed.pt",
                out_dir / "pyg_lrgb" / "peptides-func" / "processed" / "train.pt",
            ]
        )
        if processed is not None and processed.name == "hf_geometric_data_processed.pt":
            records, source_size = processed_pyg_records(
                processed,
                "peptides_func",
                args.max_nodes,
                seed=seed,
                sample_cap=max(1, int(args.max_train) + int(args.max_val) + int(args.max_test)),
            )
            return split_records(records, args, seed), {
                "loader": "cached Peptides-func processed PyG data",
                "processed_path": "<repo-local-cache>",
                "source_size": source_size,
            }
        from torch_geometric.datasets import LRGBDataset

        splits = {}
        source_size = 0
        for split in ["train", "val", "test"]:
            pyg = LRGBDataset(str(out_dir / "pyg_lrgb"), name="Peptides-func", split=split)
            source_size += len(pyg)
            records = pyg_to_records(pyg, f"peptides_func_{split}", args.max_nodes)
            splits[split] = cap_list(records, getattr(args, f"max_{split}"), seed)
        return splits, {"loader": "torch_geometric.datasets.LRGBDataset(Peptides-func)", "source_size": source_size}
    raise ValueError(f"Unsupported dataset: {dataset}")


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def processed_pyg_records(
    processed_path: Path,
    prefix: str,
    max_nodes: int,
    *,
    seed: int,
    sample_cap: int | None = None,
) -> tuple[list[GraphRecord], int]:
    loaded = torch.load(processed_path, map_location="cpu", weights_only=False)
    if isinstance(loaded, tuple) and len(loaded) == 3:
        data, slices, _data_cls = loaded
    elif isinstance(loaded, tuple) and len(loaded) == 2:
        data, slices = loaded
    else:
        raise ValueError(f"Unsupported processed PyG payload in {processed_path.name}: {type(loaded)}")
    size_key = "x" if "x" in slices else "z"
    source_size = int(slices[size_key].numel() - 1)
    indices = list(range(source_size))
    if sample_cap is not None and int(sample_cap) > 0 and source_size > int(sample_cap):
        rng = random.Random(seed)
        indices = sorted(rng.sample(indices, int(sample_cap)))
    records: list[GraphRecord] = []
    skipped = 0
    for idx in indices:
        node_start = int(slices[size_key][idx])
        node_end = int(slices[size_key][idx + 1])
        num_nodes = node_end - node_start
        if num_nodes <= 0 or num_nodes > int(max_nodes):
            skipped += 1
            continue
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        edge_index = pyg_value(data, "edge_index")
        if torch.is_tensor(edge_index) and "edge_index" in slices:
            edge_start = int(slices["edge_index"][idx])
            edge_end = int(slices["edge_index"][idx + 1])
            local_edges = edge_index[:, edge_start:edge_end].detach().cpu()
            offset = 0
            if int(local_edges.numel()) > 0:
                min_node = int(local_edges.min())
                max_node = int(local_edges.max())
                if min_node >= node_start and max_node < node_end:
                    offset = node_start
            for u, v in local_edges.t().tolist():
                u, v = int(u) - offset, int(v) - offset
                if 0 <= u < num_nodes and 0 <= v < num_nodes and u != v:
                    graph.add_edge(u, v)
        record = graph_to_record(graph, f"{prefix}_{idx:06d}", prefix)
        y = pyg_value(data, "y")
        if torch.is_tensor(y) and "y" in slices:
            y_start = int(slices["y"][idx])
            y_end = int(slices["y"][idx + 1])
            if y_end > y_start:
                record.y = y[y_start:y_end].detach().cpu().reshape(-1)[0]
        z = pyg_value(data, "z")
        x_all = pyg_value(data, "x")
        if torch.is_tensor(z) and "z" in slices:
            z_start = int(slices["z"][idx])
            z_end = int(slices["z"][idx + 1])
            if z_end > z_start:
                record.node_type = z[z_start:z_end].detach().cpu().long().clamp(min=0)
        elif torch.is_tensor(x_all):
            x = x_all[node_start:node_end]
            if x.numel() > 0:
                record.node_type = x[:, 0].detach().cpu().long().clamp(min=0)
        record.metadata.update(
            {
                "source_prefix": prefix,
                "source_index": int(idx),
                "processed_path": "<repo-local-cache>",
            }
        )
        records.append(record)
    for record in records:
        record.metadata["filtered_over_max_nodes"] = skipped
    return records, source_size


def pyg_value(data: Any, key: str) -> Any:
    if isinstance(data, dict):
        return data.get(key)
    return getattr(data, key, None)


def pyg_to_records(dataset: Any, prefix: str, max_nodes: int) -> list[GraphRecord]:
    records: list[GraphRecord] = []
    skipped = 0
    for i, data in enumerate(dataset):
        num_nodes = int(getattr(data, "num_nodes", 0) or 0)
        if num_nodes <= 0 or num_nodes > int(max_nodes):
            skipped += 1
            continue
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        edge_index = getattr(data, "edge_index", None)
        if edge_index is not None and int(edge_index.numel()) > 0:
            for u, v in edge_index.t().tolist():
                u, v = int(u), int(v)
                if u != v:
                    graph.add_edge(u, v)
        record = graph_to_record(graph, f"{prefix}_{i:06d}", prefix)
        y = getattr(data, "y", None)
        if y is not None:
            record.y = y.detach().cpu().reshape(-1)[0].long() if torch.is_tensor(y) and y.numel() else None
        x = getattr(data, "x", None)
        if torch.is_tensor(x) and x.size(0) == num_nodes and x.numel() > 0:
            record.node_type = x[:, 0].detach().cpu().long().clamp(min=0)
        record.metadata.update({"source_prefix": prefix, "source_index": i})
        records.append(record)
    for r in records:
        r.metadata["filtered_over_max_nodes"] = skipped
    return records


def cora_ego_records(data: Any, samples: int, radius: int, max_nodes: int, seed: int) -> list[GraphRecord]:
    num_nodes = int(data.num_nodes)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for u, v in data.edge_index.t().tolist():
        u, v = int(u), int(v)
        if u != v:
            graph.add_edge(u, v)
    rng = random.Random(seed)
    centers = list(range(num_nodes))
    rng.shuffle(centers)
    records: list[GraphRecord] = []
    for center in centers[: int(samples)]:
        nodes = nx.single_source_shortest_path_length(graph, center, cutoff=int(radius)).keys()
        sub_nodes = list(nodes)
        if len(sub_nodes) > int(max_nodes):
            sub_nodes = sub_nodes[: int(max_nodes)]
        sub = graph.subgraph(sub_nodes).copy()
        record = graph_to_record(sub, f"cora_ego_{center:05d}", "cora_ego")
        y = getattr(data, "y", None)
        if torch.is_tensor(y):
            record.y = y[int(center)].detach().cpu().long()
        record.metadata.update({"source_prefix": "cora", "center": int(center), "ego_radius": int(radius)})
        records.append(record)
    return records


def split_records(records: list[GraphRecord], args: argparse.Namespace, seed: int) -> dict[str, list[GraphRecord]]:
    rng = random.Random(seed)
    rows = list(records)
    rng.shuffle(rows)
    train = cap_list(rows[: max(1, int(0.7 * len(rows)))], args.max_train, seed)
    rem = rows[max(1, int(0.7 * len(rows))) :]
    val_source = rem[: max(1, int(0.333 * len(rem)))] if rem else []
    test_source = rem[max(1, int(0.333 * len(rem))) :] if rem else []
    return {
        "train": cap_list(train, args.max_train, seed),
        "val": cap_list(val_source, args.max_val, seed),
        "test": cap_list(test_source, args.max_test, seed),
    }


def cap_splits(splits: dict[str, list[GraphRecord]], args: argparse.Namespace, seed: int) -> dict[str, list[GraphRecord]]:
    return {split: cap_list(rows, getattr(args, f"max_{split}"), seed) for split, rows in splits.items()}


def cap_list(rows: list[GraphRecord], cap: int, seed: int) -> list[GraphRecord]:
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[: min(len(rows), int(cap))]


def fit_component_tokenizer_staged(records: list[GraphRecord], out_dir: Path) -> GPTok2Tokenizer:
    stage_times: dict[str, float] = {}
    tokenizer = GPTok2Tokenizer()
    t0 = time.time()
    print(f"[fit] patches_for_records: {len(records)} train graphs", flush=True)
    patch_map = patches_for_records(records, tokenizer.config)
    train_patches = [p for patches in patch_map.values() for p in patches]
    stage_times["patches_sec"] = time.time() - t0
    print(f"[fit] patches done: {len(train_patches)} patches in {stage_times['patches_sec']:.1f}s", flush=True)

    t0 = time.time()
    tokenizer.codebook = learn_codebook(train_patches, tokenizer.config)
    stage_times["codebook_sec"] = time.time() - t0
    print(f"[fit] codebook done: {len(tokenizer.codebook)} codes in {stage_times['codebook_sec']:.1f}s", flush=True)

    t0 = time.time()
    tokenizer._fit_patch_map = patch_map  # noqa: SLF001
    programs = [compile_program(r, patch_map[r.graph_id], tokenizer.codebook, tokenizer.config) for r in records]
    program_tokens = [p.to_tokens() for p in programs]
    stage_times["programs_sec"] = time.time() - t0
    print(f"[fit] programs done: {len(programs)} programs in {stage_times['programs_sec']:.1f}s", flush=True)

    compact_cfg = tokenizer.config.get("compact_entropy", {})
    motif_cfg = tokenizer.config.get("motif_macro", {})

    t0 = time.time()
    tokenizer.compact_codec = CompactCodec(
        max_macros=int(compact_cfg.get("max_macros", 384)),
        min_macro_count=int(compact_cfg.get("min_macro_count", 3)),
        max_macro_len=int(compact_cfg.get("max_macro_len", 12)),
        max_bpe_merges=int(compact_cfg.get("max_bpe_merges", 384)),
        min_bpe_count=int(compact_cfg.get("min_bpe_count", 3)),
    ).fit(program_tokens)
    compact_sequences = [tokenizer.compact_codec.encode(seq) for seq in program_tokens]
    tokenizer.compact_vocab_size = max(2, len({tok for seq in compact_sequences for tok in seq}))
    tokenizer.entropy_model = EntropyModel().fit(compact_sequences, tokenizer.compact_codec.profile)
    stage_times["compact_sec"] = time.time() - t0
    print(f"[fit] compact done in {stage_times['compact_sec']:.1f}s", flush=True)

    t0 = time.time()
    tokenizer.motif_codec = MotifMacroCodec(
        tokenizer.codebook,
        max_structural_macros=int(motif_cfg.get("max_structural_macros", 384)),
        min_structural_count=int(motif_cfg.get("min_structural_count", 2)),
        max_structural_len=int(motif_cfg.get("max_structural_len", 24)),
        max_compact_macros=int(compact_cfg.get("max_macros", 384)),
        min_compact_macro_count=int(compact_cfg.get("min_macro_count", 3)),
        max_compact_macro_len=int(compact_cfg.get("max_macro_len", 12)),
        max_bpe_merges=int(compact_cfg.get("max_bpe_merges", 384)),
        min_bpe_count=int(compact_cfg.get("min_bpe_count", 3)),
        use_parameterized_macros=bool(motif_cfg.get("use_parameterized_macros", True)),
        min_parameterized_emit_count=int(motif_cfg.get("min_parameterized_emit_count", 3)),
        max_parameterized_span_len=int(motif_cfg.get("max_parameterized_span_len", 96)),
        max_merge_schemas=int(motif_cfg.get("max_merge_schemas", 256)),
        min_merge_schema_count=int(motif_cfg.get("min_merge_schema_count", 2)),
        max_code_schemas=int(motif_cfg.get("max_code_schemas", 256)),
        min_code_schema_count=int(motif_cfg.get("min_code_schema_count", 2)),
    ).fit(program_tokens)
    motif_sequences = [tokenizer.motif_codec.encode(seq) for seq in program_tokens]
    tokenizer.motif_vocab_size = max(2, len({tok for seq in motif_sequences for tok in seq}))
    tokenizer.motif_entropy_model = MotifEntropyModel().fit(motif_sequences, tokenizer.motif_codec.profile)
    motif_hybrid_sequences = [
        tokenizer.motif_codec.encode_hybrid(seq, tokenizer.entropy_model, tokenizer.motif_entropy_model)
        for seq in program_tokens
    ]
    tokenizer.motif_hybrid_vocab_size = max(2, len({tok for seq in motif_hybrid_sequences for tok in seq}))
    tokenizer.motif_hybrid_entropy_model = MotifEntropyModel().fit(motif_hybrid_sequences, tokenizer.motif_codec.profile)
    stage_times["motif_sec"] = time.time() - t0
    stage_times["total_fit_sec"] = sum(stage_times.values())
    print(f"[fit] motif done in {stage_times['motif_sec']:.1f}s", flush=True)

    (out_dir / "stage_times.json").write_text(json.dumps(stage_times, indent=2), encoding="utf-8")
    return tokenizer


def run_dataset(dataset: str, splits: dict[str, list[GraphRecord]], out_dir: Path) -> list[dict[str, Any]]:
    start = time.time()
    tokenizer_path = out_dir / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = GPTok2Tokenizer.load(tokenizer_path)
        print(f"[fit] reuse tokenizer: {tokenizer_path}", flush=True)
    else:
        tokenizer = fit_component_tokenizer_staged(splits["train"], out_dir)
        tokenizer.save(tokenizer_path)
    rows: list[dict[str, Any]] = [
        {
            "dataset": dataset,
            "split": "dataset",
            "mode": "dataset",
            "train_graphs": len(splits["train"]),
            "val_graphs": len(splits["val"]),
            "test_graphs": len(splits["test"]),
            "graphs": sum(len(v) for v in splits.values()),
        }
    ]
    for split in ["train", "val", "test"]:
        per_rows: list[dict[str, Any]] = []
        records = splits[split]
        split_start = time.time()
        print(f"[compile] {dataset}/{split}: proposing patches for {len(records)} graphs", flush=True)
        patch_map = patches_for_records(records, tokenizer.config)
        assert tokenizer.codebook is not None
        for idx, record in enumerate(records, 1):
            program = compile_program(record, patch_map[record.graph_id], tokenizer.codebook, tokenizer.config)
            per_rows.extend(evaluate_program(tokenizer, record, program, dataset, split))
            if idx % 50 == 0 or idx == len(records):
                print(f"[progress] {dataset}/{split}: {idx}/{len(records)}", flush=True)
                write_csv(out_dir / "per_graph" / f"{split}.csv", per_rows)
        rows.extend(summarize(dataset, split, per_rows))
        print(f"[split] {dataset}/{split}: {len(records)} graphs in {time.time() - split_start:.1f}s", flush=True)
    rows[0]["elapsed_sec"] = time.time() - start
    return rows


def evaluate_program(tokenizer: GPTok2Tokenizer, record: GraphRecord, program: GraphProgram, dataset: str, split: str) -> list[dict[str, Any]]:
    original = program.to_tokens()
    assert tokenizer.codebook is not None
    assert tokenizer.compact_codec is not None and tokenizer.entropy_model is not None
    assert tokenizer.motif_codec is not None
    assert tokenizer.motif_entropy_model is not None and tokenizer.motif_hybrid_entropy_model is not None
    edges = max(1, len(edge_set(record)))
    base_fid: dict[str, float]
    try:
        base_recon, base_state = execute_program(program, tokenizer.codebook, tokenizer.config)
        base_fid = structure_fidelity(record, base_recon)
        base_fid["strict_exact_reconstruction"] = strict_isomorphic(record, base_recon)
        base_executable = float(base_state.illegal_actions == 0 and base_state.runtime_errors == 0)
    except Exception:
        base_fid = {
            "edge_f1": 0.0,
            "edge_precision": 0.0,
            "edge_recall": 0.0,
            "exact_reconstruction": 0.0,
            "strict_exact_reconstruction": 0.0,
        }
        base_executable = 0.0
    rows = []
    for mode in MODES:
        try:
            if mode == "original":
                tokens = original
                bits = original_program_bits(
                    original,
                    codebook_size=len(tokenizer.codebook),
                    ref_window=int(tokenizer.config.get("program", {}).get("ref_window", 768)),
                )
                expanded = original
            elif mode == "compact":
                tokens = tokenizer.compact_codec.encode(original)
                bits = compact_symbol_bits(tokens, tokenizer.compact_vocab_size)
                expanded = tokenizer.compact_codec.decode(tokens)
            elif mode == "entropy":
                tokens = tokenizer.compact_codec.encode(original)
                bits = tokenizer.entropy_model.bits(tokens, include_rulebook=False)
                expanded = tokenizer.compact_codec.decode(tokens)
            elif mode == "motif_macro":
                tokens = tokenizer.motif_codec.encode(original)
                bits = motif_symbol_bits(tokens, tokenizer.motif_vocab_size)
                expanded = tokenizer.motif_codec.decode(tokens)
            elif mode == "motif_entropy":
                tokens = tokenizer.motif_codec.encode(original)
                bits = tokenizer.motif_entropy_model.bits(tokens, include_rulebook=False)
                expanded = tokenizer.motif_codec.decode(tokens)
            elif mode == "motif_hybrid":
                tokens = tokenizer.motif_codec.encode_hybrid(original, tokenizer.entropy_model, tokenizer.motif_entropy_model)
                bits = tokenizer.motif_hybrid_entropy_model.bits(tokens, include_rulebook=False)
                expanded = tokenizer.motif_codec.decode(tokens)
            else:
                raise AssertionError(mode)
            lossless = float(expanded == original)
            if lossless:
                fid = base_fid
                executable = base_executable
            else:
                fid, executable = execute_expanded(tokenizer, record, expanded)
            original_count = len(original)
            bits_per_edge = float(bits) / edges
        except Exception as exc:
            tokens = []
            original_count = 0
            bits_per_edge = 0.0
            lossless = 0.0
            fid = {
                "edge_f1": 0.0,
                "edge_precision": 0.0,
                "edge_recall": 0.0,
                "exact_reconstruction": 0.0,
                "strict_exact_reconstruction": 0.0,
            }
            executable = 0.0
            error = repr(exc)
        else:
            error = ""
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "graph_id": record.graph_id,
                "mode": mode,
                "num_nodes": record.num_nodes,
                "num_edges": len(edge_set(record)),
                "token_count": len(tokens),
                "original_token_count": original_count,
                "bits_per_edge": bits_per_edge,
                "lossless_expand_match": lossless,
                "executable": executable,
                "macro_token_share": sum(1 for t in tokens if "_BLOCK(" in t or t.startswith("MACRO_")) / max(1, len(tokens)),
                "vocab_proxy": len(set(tokens)),
                "error": error,
                **fid,
            }
        )
    return rows


def execute_expanded(tokenizer: GPTok2Tokenizer, record: GraphRecord, expanded: list[str]) -> tuple[dict[str, float], float]:
    try:
        assert tokenizer.codebook is not None
        program = GraphProgram(record.graph_id, [parse_token(t) for t in expanded])
        recon, state = execute_program(program, tokenizer.codebook, tokenizer.config)
        fid = structure_fidelity(record, recon)
        fid["strict_exact_reconstruction"] = strict_isomorphic(record, recon)
        return fid, float(state.illegal_actions == 0 and state.runtime_errors == 0)
    except Exception:
        return {
            "edge_f1": 0.0,
            "edge_precision": 0.0,
            "edge_recall": 0.0,
            "exact_reconstruction": 0.0,
            "strict_exact_reconstruction": 0.0,
        }, 0.0


def strict_isomorphic(original: GraphRecord, reconstructed: GraphRecord) -> float:
    go = to_networkx(original).to_undirected()
    gr = to_networkx(reconstructed).to_undirected()
    if go.number_of_nodes() != gr.number_of_nodes() or go.number_of_edges() != gr.number_of_edges():
        return 0.0
    if sorted(dict(go.degree()).values()) != sorted(dict(gr.degree()).values()):
        return 0.0
    try:
        return float(nx.is_isomorphic(go, gr))
    except (nx.NetworkXException, RuntimeError, MemoryError):
        return 0.0


def summarize(dataset: str, split: str, per_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_rows:
        by_mode[row["mode"]].append(row)
    for mode in MODES:
        rows = by_mode.get(mode, [])
        tokens = []
        # Per-graph token strings are not stored to keep files compact; vocab_proxy is averaged instead.
        out.append(
            {
                "dataset": dataset,
                "split": split,
                "mode": mode,
                "graphs": len(rows),
                "avg_nodes": avg(rows, "num_nodes"),
                "avg_edges": avg(rows, "num_edges"),
                "avg_tokens": avg(rows, "token_count"),
                "avg_original_tokens": avg(rows, "original_token_count"),
                "avg_bits_per_edge": avg(rows, "bits_per_edge"),
                "avg_lossless_expand_match": avg(rows, "lossless_expand_match"),
                "avg_exact_reconstruction": avg(rows, "exact_reconstruction"),
                "avg_strict_exact_reconstruction": avg(rows, "strict_exact_reconstruction"),
                "avg_edge_f1": avg(rows, "edge_f1"),
                "avg_executable": avg(rows, "executable"),
                "avg_macro_token_share": avg(rows, "macro_token_share"),
                "avg_vocab_proxy": avg(rows, "vocab_proxy"),
                "failed_graphs": sum(1 for r in rows if r.get("error")),
                "token_rows_for_vocab": len(tokens),
            }
        )
    return out


def avg(rows: list[dict[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(key, 0.0)))
        except (TypeError, ValueError):
            pass
    return float(mean(vals)) if vals else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
