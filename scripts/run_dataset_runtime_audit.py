"""Per-dataset canonicalization/runtime audit for the 21-data corpus."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from gptok2.canonical import canonicalize
from gptok2.data.io import load_records
from gptok2.data.schema_spec import SchemaSpec
from gptok2_tokenizer import GPTok2Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="rebuttal/data_21")
    parser.add_argument("--out", default="rebuttal/experiments/E2_runtime_dataset")
    parser.add_argument("--max-graphs", type=int, default=8)
    parser.add_argument("--max-search-nodes", type=int, default=100)
    parser.add_argument("--timeout-sec", type=float, default=2.0)
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    per_graph=[]; summary=[]
    per_graph_path = out / "per_graph.csv"
    summary_path = out / "summary.csv"
    checkpoint_path = out / "checkpoint.json"
    completed = set()
    if per_graph_path.exists():
        with per_graph_path.open(encoding="utf-8", newline="") as fh:
            per_graph = list(csv.DictReader(fh))
    if summary_path.exists():
        with summary_path.open(encoding="utf-8", newline="") as fh:
            summary = list(csv.DictReader(fh))
    if checkpoint_path.exists():
        completed = set(json.loads(checkpoint_path.read_text(encoding="utf-8")).get("completed", []))
    for path in sorted(Path(args.data_dir).glob("*.jsonl")):
        name=path.stem; records=load_records(path)[:args.max_graphs]
        if not records: continue
        if name in set(args.skip):
            continue
        if name in completed:
            continue
        spec=SchemaSpec(max_search_nodes=args.max_search_nodes, timeout_sec=args.timeout_sec)
        canonical_records=[]; canonical_times=[]; search_nodes=[]; exact=[]; fallback=[]
        for record in records:
            started=time.perf_counter(); result=canonicalize(record,spec,exact=True,max_search_nodes=args.max_search_nodes,timeout_sec=args.timeout_sec); elapsed=time.perf_counter()-started
            canonical_records.append(result.record); canonical_times.append(elapsed); search_nodes.append(result.search_nodes); exact.append(result.exact_completed); fallback.append(result.fallback_used)
            per_graph.append({"dataset":name,"graph_id":record.graph_id,"nodes":record.num_nodes,"edges":int(record.edge_index.shape[1]),"canonicalization_sec":elapsed,"search_nodes":result.search_nodes,"exact_completed":result.exact_completed,"fallback_used":result.fallback_used,"tie_groups":len(result.tie_groups)})
        fit_started=time.perf_counter(); tokenizer=GPTok2Tokenizer({"canonicalization":{"enabled":False}}).fit(canonical_records); fit_sec=time.perf_counter()-fit_started
        encode_started=time.perf_counter()
        for record in canonical_records: tokenizer.encode(record, mode="motif_hybrid")
        encode_sec=time.perf_counter()-encode_started
        total=sum(canonical_times)+fit_sec+encode_sec
        summary.append({"dataset":name,"graphs":len(records),"nodes_mean":float(np.mean([r.num_nodes for r in records])),"edges_mean":float(np.mean([r.edge_index.shape[1] for r in records])),"canonicalization_sec_total":sum(canonical_times),"canonicalization_sec_median":float(np.median(canonical_times)),"canonicalization_sec_p95":float(np.percentile(canonical_times,95)),"fit_sec":fit_sec,"encode_sec":encode_sec,"total_sec":total,"canonical_fraction":sum(canonical_times)/max(total,1e-12),"search_nodes_max":max(search_nodes),"exact_fraction":float(np.mean(exact)),"fallback_fraction":float(np.mean(fallback)),"environment_python":sys.version.split()[0],"environment_platform":platform.platform()})
        for filename, rows in (("per_graph.csv",per_graph),("summary.csv",summary)):
            with (out/filename).open("w",encoding="utf-8",newline="") as fh:
                writer=csv.DictWriter(fh,fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)
        completed.add(name)
        checkpoint_path.write_text(json.dumps({"completed": sorted(completed), "max_graphs": args.max_graphs, "max_search_nodes": args.max_search_nodes, "timeout_sec": args.timeout_sec}, indent=2), encoding="utf-8")
    for filename, rows in (("per_graph.csv",per_graph),("summary.csv",summary)):
        if not rows: continue
        with (out/filename).open("w",encoding="utf-8",newline="") as fh:
            writer=csv.DictWriter(fh,fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)
    (out/"environment.json").write_text(json.dumps({"python":sys.version,"platform":platform.platform(),"max_search_nodes":args.max_search_nodes,"timeout_sec":args.timeout_sec},indent=2),encoding="utf-8")
    print(json.dumps({"datasets":len(summary),"graphs":len(per_graph),"out":str(out)},indent=2))


if __name__=="__main__": main()
