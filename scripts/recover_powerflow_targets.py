"""Attach a declared physical target to the cached IEEE power-grid records."""

from __future__ import annotations

import json
from pathlib import Path

import pandapower as pp
import pandapower.networks as pn


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {}
for name in ["case9", "case14", "case30"]:
    net = getattr(pn, name)()
    pp.runpp(net, init="flat")
    TARGETS[f"pandapower_{name}"] = {
        "mean_bus_vm_pu": float(net.res_bus.vm_pu.mean()),
        "min_bus_vm_pu": float(net.res_bus.vm_pu.min()),
        "max_bus_vm_pu": float(net.res_bus.vm_pu.max()),
        "line_loss_mw": float(net.res_line.pl_mw.sum()),
    }


def main() -> None:
    source = ROOT / "rebuttal/data_21/ieee_power_grid.jsonl"
    output = ROOT / "rebuttal/data_21_labeled/ieee_power_grid.jsonl"
    rows = []
    for line in source.read_text().splitlines():
        row = json.loads(line)
        target = TARGETS.get(row["graph_id"])
        if target is None:
            raise ValueError(f"no solved power-flow target for {row['graph_id']}")
        row["y"] = target["mean_bus_vm_pu"]
        # Prevent target leakage: vm_pu is the regression target and must not
        # remain in node features consumed by the tokenizer.
        keys = list(row.get("metadata", {}).get("node_attr_keys", []))
        if "vm_pu" in keys and isinstance(row.get("node_attr"), list):
            idx = keys.index("vm_pu")
            row["node_attr"] = [[value for j, value in enumerate(values) if j != idx] for values in row["node_attr"]]
            row["metadata"]["node_attr_keys"] = [key for key in keys if key != "vm_pu"]
        row.setdefault("metadata", {})["label_source"] = "pandapower.runpp"
        row["metadata"]["label_recovery"] = "full_case_power_flow_target"
        row["metadata"]["task_target"] = "mean_bus_vm_pu"
        row["metadata"]["target_units"] = "per_unit"
        row["metadata"]["input_target_leakage_check"] = "vm_pu_removed_from_node_attr"
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    (ROOT / "rebuttal/audits/powerflow_target_recovery.json").write_text(json.dumps({"task": "mean_bus_vm_pu_regression", "rows": rows, "targets": TARGETS}, indent=2), encoding="utf-8")
    print(json.dumps(TARGETS, indent=2))


if __name__ == "__main__": main()
