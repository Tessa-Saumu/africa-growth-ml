"""Static repo guards: no hardcoded absolute paths, no print() in production.

Encodes final-acceptance checklist items 4 and 5 (remediation plan) as tests,
so M1/M2-class regressions fail the suite instead of a reviewer's machine.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {".git", ".opencode", ".venv", "__pycache__", "node_modules",
                ".artifacts_pre", ".pytest_cache", "logs"}

# Patterns from the final-acceptance checklist: author-machine paths.
# (Executed notebook outputs may show wherever the notebook was run; the
# banned ones are the leaked Windows/home-author paths from the old cycle.)
# Built from parts so this file cannot match its own patterns.
BAD_PATH_RE = re.compile("|".join([r"C:" + chr(92) * 2 + "dev",
                                   r"C:" + chr(92) * 2 + "Users",
                                   "/Use" + "rs/[a-z]+",
                                   "in" + "gex"]))
# notebooks legitimately print; src/, scripts/, app.py must not
PRODUCTION_CODE = ["src", "scripts"]


def _iter_source_files(patterns):
    for p in REPO_ROOT.rglob("*"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.suffix in patterns and p.is_file():
            yield p


def test_no_hardcoded_absolute_paths_in_shipped_files():
    """M2: notebooks/scripts/docs must not contain machine-specific paths."""
    offenders = []
    for p in _iter_source_files({".py", ".ipynb", ".md", ".toml", ".yaml", ".txt"}):
        # ignore the gitignore itself and review/plan docs under .opencode
        if p.name in {".gitignore", Path(__file__).name}:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if BAD_PATH_RE.search(text):
            offenders.append(str(p.relative_to(REPO_ROOT)))
    assert not offenders, f"Hardcoded/absolute machine paths found in: {offenders}"


def test_no_print_in_production_code():
    """AGENTS rule 4: src/, scripts/ and app.py log, never print."""
    offenders = []
    roots = [REPO_ROOT / d for d in PRODUCTION_CODE] + [REPO_ROOT / "app.py"]
    for root in roots:
        files = [root] if root.is_file() else list(root.rglob("*.py"))
        for p in files:
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # bare print( call (not in a string doc-line heuristic)
                if re.search(r"(?<![\w.])print\(", stripped):
                    offenders.append(f"{p.relative_to(REPO_ROOT)}:{i}: {stripped[:60]}")
    assert not offenders, f"print() found in production code: {offenders}"


def test_deployment_artifacts_not_gitignored():
    """B1: committed artifacts are load-bearing and must stay tracked."""
    import subprocess
    for rel in ["models/growth_model.joblib", "data/processed/model_data.parquet"]:
        r = subprocess.run(["git", "check-ignore", "-q", rel],
                           cwd=REPO_ROOT, capture_output=True)
        assert r.returncode != 0, f"{rel} is gitignored — deployment would break"
