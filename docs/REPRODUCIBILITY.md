# Reproducibility

This repository is organized so a reviewer can run it from zero and obtain the
same published artifacts stored under `reports/`.

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

This runs the tokenizer smoke path and a small consistency check.

## 3. Full release reproduction

```bash
python scripts/reproduce_release.py
```

That command runs the final evaluation chain:

1. tokenizer smoke;
2. representative evaluation;
3. Cora full-graph evaluation;
4. release verification.

## 4. Verification

```bash
python scripts/verify_release.py
```

This confirms the core release artifacts exist, package imports work, and the
release tree passes the anonymous-review hygiene checks.

## 5. Published reference artifacts

The canonical checked-in results live in:

- `reports/representative/`
- `reports/representative/cora_full_graph_summary.csv`

These are the files to compare against after re-running the pipeline.
