# AGENTS.md

1. **Verify code:** Never claim code works without running it. After every code change, run relevant tests and report the exact verification performed. If not run, state `NOT VERIFIED`.

2. **Tests:** Every `src/*.py` file must have `tests/test_<module>.py`. Unit tests only. Every code change requires corresponding test updates.

3. **Docstrings:** Every Python file starts with a module docstring explaining its purpose. Every function has a docstring specifying input arguments/formats and output arguments/formats. Use type hints.

4. **Logging:** Use `logging`, never `print()` for operational output. Log meaningful pipeline, data, training, evaluation, artifact, warning, and error events. Never log secrets.

5. **No silent failures:** Do not silently swallow errors, unexpected data, or invalid inputs.

6. **No leakage:** Never use future information when constructing features, preprocessing, validation, or test data.

7. **Notebooks:** Keep notebooks lean. Every section must use `Markdown explanation → clearly labelled code → output → brief interpretation`. Production logic belongs in `src/`, not notebooks.

8. **Visuals:** `src/visualization.py` must use a deliberate, consistent project visual language; avoid generic/default-looking charts.

9. **Dependencies:** Do not add dependencies unless necessary. Update `requirements.txt` when one is added.

10. **Scope:** Follow the existing capstone specification. Do not introduce new architecture, frameworks, services, or unnecessary abstractions without explicit approval.

11. **Changes:** Inspect existing code and tests first, make the smallest appropriate change, update tests, run verification, and review the diff.

12. **Final status:** Every coding response must state what changed, tests run, verification result, and `VERIFIED` or `NOT VERIFIED`.

---

## Notebook Kernel Setup (avoid 30min rabbit hole)

To execute notebooks with `jupyter nbconvert --execute`, the kernel must find the project's `src/` module.

**Working approach:**
```bash
pip install -e .                    # Install package in editable mode (creates .pth in site-packages)
python -m ipykernel install --user --name=africa-growth-ml  # Register kernel
# In notebook, use absolute path for imports:
from pathlib import Path
PROJECT_ROOT = Path(r'C:\dev\africa-growth-ml')
import sys
sys.path.insert(0, str(PROJECT_ROOT))
```

**What fails:** Relative imports (`sys.path.insert(0, '../src')`) don't work because nbconvert runs from a temp dir. The kernel needs the project root in sys.path AND the package installed so `src.config` resolves.