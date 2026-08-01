# Reproducibility

This repository is organized so a reviewer can run the tokenizer, downstream
Transformer consumer, and evaluation pipeline from this directory. Datasets are
downloaded or supplied externally; code and checked-in reference artifacts are
kept in the repository.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-cu124.txt
python -m pip install -e .
```

If you do not want the exact `cu124` torch wheel, use `requirements.txt`
instead. For the representative dataset loaders:

```bash
python -m pip install -r requirements-full.txt
```

## 2. Smoke check

```bash
python scripts/reproduce_smoke.py
```

This runs the tokenizer smoke path, the downstream Transformer consumer smoke
path, and the unit checks.

## 3. Downstream Transformer consumer

Prepare tokenized graph examples from a GraphRecord JSONL file:

```bash
python -m rps_gtok_consumption.cli prepare \
  --input path/to/graphs.jsonl \
  --output runs/prepared/tokenized.jsonl \
  --mode motif_hybrid \
  --task-type classification
```

Train the Full-Embed consumer with the shared Transformer backbone:

```bash
python -m rps_gtok_consumption.cli train \
  --train path/to/train.jsonl \
  --val path/to/val.jsonl \
  --test path/to/test.jsonl \
  --config configs/consumer_full_embed.yaml \
  --out runs/rps_gtok_consumer
```

Run the matched token-view experiment runner:

```bash
python -m rps_gtok_consumption.cli run-config \
  --config configs/consumer_multiview_smoke.yaml \
  --out runs/rps_gtok_multiview
```

The matched-view runner supports RPS-GTok full, compact, atomic-program,
shuffled, random-ID, edge-list BPE, adjacency-list BPE, DFS/BFS BPE, and
frequency-guided walk BPE views under the same training loop.

## 4. Full release reproduction

```bash
python scripts/reproduce_release.py
```

That command runs the final evaluation chain:

1. tokenizer smoke;
2. representative evaluation;
3. Cora full-graph evaluation;
4. release verification.

## 5. Verification

```bash
python scripts/verify_release.py
```

This confirms the core release artifacts exist, package imports work, and the
release tree passes the anonymous-review hygiene checks.

## 6. Published reference artifacts

The canonical checked-in results live in:

- `reports/representative/`
- `reports/representative/cora_full_graph_summary.csv`

These are the files to compare against after re-running the pipeline.
