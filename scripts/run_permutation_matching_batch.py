"""Run permutation-matching protocol for a dataset subset with checkpoints."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument("--data-dir",default="rebuttal/data_21");p.add_argument("--out",required=True);p.add_argument("--datasets",nargs="+",required=True);p.add_argument("--device",default="cuda");p.add_argument("--epochs",type=int,default=12);p.add_argument("--max-graphs",type=int,default=60);p.add_argument("--exact-only",action="store_true");p.add_argument("--canonical-max-search",type=int,default=1000);p.add_argument("--canonical-timeout-sec",type=float,default=.5);a=p.parse_args(); root=Path(a.out); root.mkdir(parents=True,exist_ok=True)
    for dataset in a.datasets:
        run=root/dataset
        if (run/"results.csv").exists(): continue
        cmd=[sys.executable,"scripts/run_permutation_matching_downstream.py","--data-dir",a.data_dir,"--dataset",dataset,"--out",str(run),"--views","edge_list","canonical_edge_list","rps_gtok_plus","--max-graphs",str(a.max_graphs),"--seeds","2026","2027","2028","--epochs",str(a.epochs),"--batch-size","32","--device",a.device,"--canonical-max-search",str(a.canonical_max_search),"--canonical-timeout-sec",str(a.canonical_timeout_sec)]
        if a.exact_only: cmd.append("--exact-only")
        result=subprocess.run(cmd,check=False)
        (root/"status.log").open("a",encoding="utf-8").write(f"{dataset}\t{result.returncode}\n")
        if result.returncode != 0: continue
    print({"datasets":a.datasets,"out":str(root)})

if __name__=="__main__":main()
