"""Bounded, independently certified permutation audit using pynauty.

Unlike the research canonicalizer's bounded individualization, this audit
uses nauty's colored incidence-graph certificate.  Each graph is isolated in
a child process, so a hard symmetric case is recorded as a timeout rather
than hanging the release pipeline or being silently treated as exact.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pynauty
import torch

from gptok2.data.io import load_records
from gptok2.data.schema import record_from_dict
from rps_gtok_consumption.exact_backend import exact_certificate
from scripts.run_topology_shift_downstream import permute_record


def _row_tensor(t: torch.Tensor | None, i: int) -> tuple:
    if t is None or i >= t.shape[0]:
        return ()
    return tuple(t[i].detach().cpu().reshape(-1).tolist())


def _worker(row: dict, seed: int, queue) -> None:
    try:
        record = record_from_dict(row)
        other = permute_record(record, seed)
        left = exact_certificate(record)
        right = exact_certificate(other)
        queue.put({"status": "ok", "equal": bool(left == right), "nodes": int(record.num_nodes), "edges": int(record.edge_index.shape[1])})
    except Exception as exc:  # pragma: no cover - worker boundary
        queue.put({"status": "error", "error": repr(exc)})


def _one(row: dict, seed: int, timeout: float) -> dict:
    # ``fork`` from a multithreaded scheduler can inherit libgmp/pynauty
    # locks and produce spurious hangs.  ``spawn`` gives each exact job a
    # clean interpreter while preserving the hard kill boundary.
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(row, seed, queue), daemon=True)
    proc.start(); proc.join(timeout)
    if proc.is_alive():
        proc.terminate(); proc.join()
        return {"status": "timeout"}
    return queue.get() if not queue.empty() else {"status": "error", "error": "worker_no_result"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("rebuttal/data_21"))
    p.add_argument("--out", type=Path, default=Path("rebuttal/tables/Table_E11_nauty_exact_permutation_audit.csv"))
    p.add_argument("--max-graphs", type=int, default=60,
                   help="maximum graphs per dataset; default covers the materialized corpus")
    p.add_argument("--timeout-sec", type=float, default=10.0,
                   help="hard per-graph subprocess budget")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                   help="concurrent isolated graph workers")
    p.add_argument("--datasets", nargs="*", default=None)
    args = p.parse_args()
    rows = []
    for path in sorted(args.data_dir.glob("*.jsonl")):
        if args.datasets is not None and path.stem not in set(args.datasets):
            continue
        records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()][: args.max_graphs]
        counts = {"ok": 0, "equal": 0, "timeout": 0, "error": 0, "directed": 0}
        directed_input = sum(int(bool(row.get("directed", False))) for row in records)
        node_max = edge_max = 0
        started = time.perf_counter()
        # The parent uses threads only as a scheduler.  Every actual nauty
        # invocation remains in its own killable subprocess, so a hard case
        # cannot consume a worker indefinitely.
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(_one, row, 910000 + i, args.timeout_sec): i
                for i, row in enumerate(records)
            }
            for future in as_completed(futures):
                result = future.result()
                status = result.get("status", "error")
                counts[status] = counts.get(status, 0) + 1
                if status == "ok":
                    counts["equal"] += int(result.get("equal", False))
                    node_max = max(node_max, int(result.get("nodes", 0)))
                    edge_max = max(edge_max, int(result.get("edges", 0)))
        rows.append({"dataset": path.stem, "graphs": len(records), **counts, "directed_input": directed_input, "certified_equal_rate": counts["equal"] / max(1, counts["ok"]), "wall_sec": time.perf_counter() - started, "timeout_sec": args.timeout_sec, "workers": args.workers, "certificate": "pynauty_attributed_incidence_graph_v2"})
        print(rows[-1], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.out.with_suffix(".protocol.json")).write_text(json.dumps({"exact_backend": "pynauty", "certificate_encoding": "direct_simple_or_wl_refined_attributed_incidence_v3", "direct_simple_graph_fast_path": True, "undirected_reciprocal_coo_is_one_semantic_edge": True, "per_graph_timeout_sec": args.timeout_sec, "workers": args.workers, "max_graphs": args.max_graphs, "no_label_or_task_claim": True}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
