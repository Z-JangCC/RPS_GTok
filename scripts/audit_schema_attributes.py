"""Materialize the reconstruction target and attribute policy per dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from gptok2.data.io import load_records


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-dir", default="rebuttal/data_21"); parser.add_argument("--manifest", default="configs/schema_manifest.yaml"); parser.add_argument("--out", default="rebuttal/tables/Table_R10_schema_attributes.csv")
    args = parser.parse_args()
    manifest = {}
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        try:
            import yaml
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        except Exception:
            manifest = {}
    rows=[]
    for path in sorted(Path(args.data_dir).glob("*.jsonl")):
        records=load_records(path)
        if not records: continue
        node_attr_dims=sorted({int(r.node_attr.shape[1]) if r.node_attr is not None and r.node_attr.ndim > 1 else 0 for r in records})
        edge_attr_dims=sorted({int(r.edge_attr.shape[1]) if r.edge_attr is not None and r.edge_attr.ndim > 1 else 0 for r in records})
        node_types=any(r.node_type is not None for r in records); edge_types=any(r.edge_type is not None for r in records)
        cfg=manifest.get(path.stem, manifest.get("datasets", {}).get(path.stem, {})) if isinstance(manifest, dict) else {}
        quant=cfg.get("quantization", {}) if isinstance(cfg, dict) else {}
        node_dtypes=sorted({str(r.node_attr.dtype) for r in records if r.node_attr is not None})
        edge_dtypes=sorted({str(r.edge_attr.dtype) for r in records if r.edge_attr is not None})
        node_unique=sorted({int(r.node_attr.reshape(-1).unique().numel()) for r in records if r.node_attr is not None})
        edge_unique=sorted({int(r.edge_attr.reshape(-1).unique().numel()) for r in records if r.edge_attr is not None})
        rows.append({"dataset":path.stem,"graphs":len(records),"directed":any(bool(r.directed) for r in records),"node_attr_dims":json.dumps(node_attr_dims),"edge_attr_dims":json.dumps(edge_attr_dims),"node_attr_dtypes":json.dumps(node_dtypes),"edge_attr_dtypes":json.dumps(edge_dtypes),"node_attr_unique_values":json.dumps(node_unique),"edge_attr_unique_values":json.dumps(edge_unique),"node_type_present":node_types,"edge_type_present":edge_types,"continuous_mode":quant.get("mode",cfg.get("continuous_mode","raw") if isinstance(cfg,dict) else "raw"),"quantization_decimals":quant.get("decimals",cfg.get("quantization_decimals") if isinstance(cfg,dict) else None),"reconstruction_target":"topology+schema fields; continuous attributes are retained under the stated mode"})
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__=="__main__": main()
