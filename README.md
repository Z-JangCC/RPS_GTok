# RPS_GTok

`RPS_GTok` is a self-contained release of the final RPS-GTok graph tokenizer
and downstream Transformer consumer code used for the paper experiments,
prepared for reproducible sharing and continued development from this directory.

## What is included

- `gptok2_tokenizer/`: main tokenizer API and CLI.
- `rps_gtok_consumption/`: Full-Embed token adapter, shared Transformer
  backbone, tokenized graph datasets, and train/evaluate CLI.
- `gptok2_evaluation/`: final representative dataset and Cora full-graph evaluation.
- `gptok2_compact/`, `gptok2_motif_macro/`, `gptok2/`: supporting code
  required by the final tokenizer.
- `reports/`: checked-in final reference CSV results to compare against.
- `tests/`: smoke tests.
- `scripts/`: one-command reproduction and verification helpers.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-cu124.txt
python -m pip install -e .
```

If you do not want the exact `cu124` torch wheel, use `requirements.txt`
instead. For the representative loaders used by the full evaluation:

```bash
python -m pip install -r requirements-full.txt
```

## Run

Smoke check:

```bash
python scripts/reproduce_smoke.py
```

Synthetic downstream Transformer check:

```bash
python -m rps_gtok_consumption.cli smoke
```

Train a consumer from prepared tokenized examples:

```bash
python -m rps_gtok_consumption.cli train \
  --train path/to/train.jsonl \
  --val path/to/val.jsonl \
  --test path/to/test.jsonl \
  --config configs/consumer_full_embed.yaml \
  --out runs/rps_gtok_consumer
```

Full reproduction:

```bash
python scripts/reproduce_release.py
```

Verification:

```bash
python scripts/verify_release.py
```

## Primary artifacts

- `reports/representative/summary_all.csv`
- `reports/representative/test_comparison.csv`
- `reports/representative/dataset_status.csv`
- `reports/representative/cora_full_graph_summary.csv`

## Notes

- The repository is designed to run from the project root.
- The downstream consumer code can be run with supplied GraphRecord JSONL files
  or with prepared tokenized JSONL files; datasets themselves are not vendored.
- The checked-in report files are the canonical release snapshots.
- First-time full evaluation may download TU Dortmund, PyG, or OGB datasets.
- The reference environment used Python 3.12, PyTorch 2.5.1+cu124, and CUDA 12.4.
- This release is published under the MIT License.

For the exact command sequence, see `docs/REPRODUCIBILITY.md`.
