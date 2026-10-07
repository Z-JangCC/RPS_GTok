"""Matched message-passing reference baseline for semantic graph tasks.

This is an external reference, not a tokenizer competitor. It uses the same
graph split, seeds, class weighting, validation selection and test protocol as
the sequence consumers so claims about RPS are not compared only against weak
serialization baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

from gptok2.data.io import load_records


class GINClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, classes: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for index in range(3):
            inp = in_dim if index == 0 else hidden
            self.layers.append(GINConv(nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden))))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, classes))

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        for layer in self.layers:
            x = torch.relu(layer(x, data.edge_index))
        return self.head(global_add_pool(x, data.batch))


def to_data(record) -> Data:
    n = int(record.num_nodes)
    degree = torch.zeros(n, 1, dtype=torch.float32)
    if record.edge_index.numel():
        for node in record.edge_index.reshape(-1).tolist():
            degree[int(node), 0] += 1.0
    x = torch.cat([degree, torch.ones(n, 1)], dim=1)
    edge_index = record.edge_index.clone().long()
    if not record.directed and edge_index.numel():
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([int(record.y.item())], dtype=torch.long), graph_id=record.graph_id)


def evaluate(model, loader, device):
    model.eval(); truth=[]; pred=[]
    with torch.no_grad():
        for batch in loader:
            out=model(batch.to(device)); pred.extend(out.argmax(-1).cpu().tolist()); truth.extend(batch.y.cpu().tolist())
    return {
        "accuracy": float(np.mean(np.asarray(pred)==np.asarray(truth))) if truth else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)) if len(set(truth)) > 1 else 0.0,
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)) if truth else 0.0,
        "prediction_unique": int(len(set(pred))),
    }


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--train",required=True); parser.add_argument("--val",required=True); parser.add_argument("--test",required=True); parser.add_argument("--out",required=True); parser.add_argument("--seeds",nargs="+",type=int,default=[2026,2027,2028]); parser.add_argument("--epochs",type=int,default=40); parser.add_argument("--device",default="auto"); parser.add_argument("--max-train",type=int,default=0); parser.add_argument("--max-val",type=int,default=0); parser.add_argument("--max-test",type=int,default=0)
    args=parser.parse_args(); device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else args.device if args.device!="auto" else "cpu")
    splits={k:[to_data(r) for r in load_records(v)] for k,v in {"train":args.train,"val":args.val,"test":args.test}.items()}
    for key, limit in (("train", args.max_train), ("val", args.max_val), ("test", args.max_test)):
        if limit > 0:
            splits[key] = splits[key][:limit]
    labels=[int(d.y.item()) for d in splits["train"]]; classes=max(labels)+1; counts=np.bincount(labels,minlength=classes); weights=torch.tensor(len(labels)/(classes*np.maximum(1,counts)),dtype=torch.float32,device=device)
    rows=[]
    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model=GINClassifier(2,64,classes).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-2); criterion=nn.CrossEntropyLoss(weight=weights); best=None; stale=0
        loaders={k:DataLoader(v,batch_size=16,shuffle=k=="train") for k,v in splits.items()}
        for epoch in range(args.epochs):
            model.train()
            for batch in loaders["train"]:
                batch=batch.to(device); opt.zero_grad(); loss=criterion(model(batch),batch.y); loss.backward(); opt.step()
            val=evaluate(model,loaders["val"],device); score=(val["balanced_accuracy"],val["macro_f1"])
            if best is None or score>best[0]: best=(score,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}); stale=0
            else: stale+=1
            if stale>=8: break
        model.load_state_dict(best[1]); test=evaluate(model,loaders["test"],device); rows.append({"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),**test})
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); fields=sorted({k for r in rows for k in r});
    with out.open("w",encoding="utf-8",newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows":len(rows),"device":str(device),"parameters":rows[0]["parameters"]},indent=2))


if __name__=="__main__": main()
