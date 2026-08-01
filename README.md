# RPS_GTok

`RPS_GTok` is a self-contained release of the final RPS-GTok graph tokenizer,
the downstream Transformer consumer, and the matched token-view experiments
used for the paper. The repository is intended to work as the only development
directory: no code outside this folder is required, except for datasets that are
downloaded or supplied separately.

## Contents

- `gptok2_tokenizer/`: final tokenizer API and CLI.
- `gptok2/`: graph schema, patch proposal, VQ codebook, graph-program compiler,
  interpreter, and reconstruction metrics.
- `gptok2_compact/`: reversible compact and entropy coding layers.
- `gptok2_motif_macro/`: final motif/macro coding layer.
- `rps_gtok_consumption/`: Full-Embed token adapter, shared Transformer
  backbone, matched token-view builders, prepared-token datasets, and training
  CLI.
- `gptok2_evaluation/`: representative dataset and Cora full-graph evaluation
  scripts.
- `configs/`: runnable consumer configurations.
- `reports/`: checked-in reference CSV artifacts.
- `tests/`, `scripts/`, `examples/`: smoke tests, release verification, and
  small usage examples.

## Installation

The reference environment used Python 3.12, PyTorch 2.5.1+cu124, and CUDA 12.4.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-cu124.txt
python -m pip install -e .
```

If an exact CUDA 12.4 torch wheel is not needed, use:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

For full dataset loaders used by the representative evaluation:

```bash
python -m pip install -r requirements-full.txt
```

## Quick Checks

Run the complete local smoke path:

```bash
python scripts/reproduce_smoke.py
```

Run the release and anonymity audit:

```bash
python scripts/verify_release.py
```

Expected smoke behavior:

- tokenizer round-trip and reconstruction checks pass;
- Full-Embed Transformer consumer trains on synthetic graphs;
- unit tests pass.

## Tokenizer Usage

Python API:

```python
from gptok2_tokenizer import GPTok2Tokenizer
from gptok2.data.synthetic import generate_synthetic_graphs

config = {"data": {"synthetic": {"num_graphs": 24}}}
graphs = generate_synthetic_graphs(config, seed=2026)
tokenizer = GPTok2Tokenizer().fit(graphs)
encoded = tokenizer.encode(graphs[0], mode="motif_hybrid")
reconstructed = tokenizer.decode(encoded)
```

CLI:

```bash
python -m gptok2_tokenizer.cli smoke --out runs/rps_gtok_smoke
python -m gptok2_tokenizer.cli fit --input path/to/graphs.jsonl --artifact runs/tokenizer.json
python -m gptok2_tokenizer.cli encode --artifact runs/tokenizer.json --input path/to/graphs.jsonl --output runs/tokens.jsonl --mode all
```

Supported tokenizer modes:

- `original`
- `compact`
- `entropy`
- `motif_macro`
- `motif_entropy`
- `motif_hybrid`

## Downstream Transformer Consumer

The downstream consumer uses:

`token IDs -> token adapter -> shared Transformer backbone -> pooled graph head`

The Full-Embed variant changes the token adapter before the Transformer by
adding learned embeddings for token kind, segment, token-id bins, span bins,
arity bins, reference-count bins, and reference roles. The Transformer backbone
and graph-level prediction head are shared by the plain and Full-Embed
variants.

Run a synthetic Full-Embed smoke experiment:

```bash
python -m rps_gtok_consumption.cli smoke --out runs/rps_gtok_consumer_smoke
```

Prepare tokenized examples from external `GraphRecord` JSONL data:

```bash
python -m rps_gtok_consumption.cli prepare \
  --input path/to/graphs.jsonl \
  --output runs/prepared/tokenized.jsonl \
  --mode motif_hybrid \
  --task-type classification
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

## Matched Token-View Experiments

The config-driven runner trains the same consumer loop over matched token views:

```bash
python -m rps_gtok_consumption.cli run-config \
  --config configs/consumer_multiview_smoke.yaml \
  --out runs/rps_gtok_multiview
```

Implemented views:

- `rps_gtok_full`
- `rps_gtok_compact`
- `atomic_program`
- `rps_gtok_shuffled`
- `rps_gtok_random_ids`
- `edge_list_bpe`
- `adjacency_list_bpe`
- `dfs_order_bpe`
- `bfs_order_bpe`
- `graph_tokenizer_feuler_bpe`

The BPE views fit merge rules on the training split only, then apply the frozen
view encoder to validation and test splits.

## Full Reproduction

```bash
python scripts/reproduce_release.py
```

This runs:

1. tokenizer smoke;
2. downstream consumer smoke;
3. representative dataset evaluation;
4. Cora full-graph evaluation;
5. release verification.

First-time full evaluation may download TU Dortmund, PyG, or OGB datasets.
Datasets are not vendored in this repository.

## Reference Artifacts

Checked-in reference results:

- `reports/representative/summary_all.csv`
- `reports/representative/test_comparison.csv`
- `reports/representative/dataset_status.csv`
- `reports/representative/cora_full_graph_summary.csv`

Use these files as release snapshots when comparing a fresh run.

## Development Notes

- Run all commands from the repository root.
- Generated outputs should stay under `runs/`, which is ignored by git.
- The anonymous release audit checks for cache files, run logs, local paths,
  process-version artifacts, personal identifiers, and secret-like tokens.
- Git commit authors in the release history are anonymized.
- The project is licensed under the MIT License.

For the exact command sequence and additional reproduction notes, see
`docs/REPRODUCIBILITY.md`.
