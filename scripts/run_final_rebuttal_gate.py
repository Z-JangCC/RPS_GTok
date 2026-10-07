"""Run the final rebuttal evidence, code, link and manuscript gates."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command(label: str, args: list[str], cwd: Path = ROOT) -> dict:
    proc = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"label": label, "returncode": proc.returncode, "output_tail": proc.stdout[-4000:]}


def link_audit() -> dict:
    files = [ROOT / "rebuttal/FINAL_REBUTTAL_RESPONSE_DRAFT.md", ROOT / "rebuttal/reviewer_responses/rebuttal_summary.md"]
    missing=[]
    for path in files:
        text=path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://")):
                continue
            resolved=(path.parent/target).resolve()
            if not resolved.exists(): missing.append({"file":str(path.relative_to(ROOT)),"target":target})
    return {"ok": not missing, "missing": missing}


def clear_python_caches() -> None:
    """Remove only regenerated interpreter caches before release hygiene."""
    for path in ROOT.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path)
    for path in ROOT.rglob("*.pyc"):
        if path.is_file():
            path.unlink()


def main() -> None:
    checks=[
        command("evidence_audit", ["python", "scripts/audit_all_rebuttal_results.py"]),
        command("matched_consumer_audit", ["python", "scripts/audit_matched_consumer_runs.py"]),
        command("pytest", ["python", "-m", "pytest", "-q"]),
    ]
    clear_python_caches()
    checks.extend([
        command("release_verify", ["python", "scripts/verify_release.py"]),
        command("number_manifest", ["python", "scripts/build_rebuttal_number_manifest.py"]),
    ])
    links=link_audit()
    manuscript=ROOT.parent / "KDD_paper/graph_tokenizer_kdd_package/acm_submission/main_four.pdf"
    report={"checks":checks,"links":links,"manuscript_pdf":{"path":str(manuscript),"exists":manuscript.exists()},"all_commands_pass":all(item["returncode"]==0 for item in checks) and links["ok"] and manuscript.exists()}
    out=ROOT / "rebuttal/final_rebuttal_gate_report.json"
    out.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))
    if not report["all_commands_pass"]: raise SystemExit(1)


if __name__ == "__main__": main()
