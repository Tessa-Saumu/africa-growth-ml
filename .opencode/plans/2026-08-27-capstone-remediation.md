# Africa Growth Explorer — Remediation Plan (v4)

> **For agentic workers:** Implement task-by-task. Every task has explicit acceptance criteria and a verification command. Do not mark a task complete until its verification command passes and you have pasted the output.

**Goal:** Close every finding in `.opencode/reviews/2026-08-27-capstone-review-v1.md` so the submission clears both the AnalystLab internship brief and the FlyRank enhanced plan (`CAPSTONE_SPEC.md`), with no outstanding fixes at final review.

**Source of truth for findings:** `.opencode/reviews/2026-08-27-capstone-review-v1.md` (C1–C6, H1–H5, M1–M12).

**Prime directive:** *Honesty over optimism.* Several findings exist because the previous cycle reported results it had not verified. Any number that appears in a document must be generated from a committed artifact by a script. If a result is unflattering, report it plainly and interpret it well. **A defensible null result scores higher than an undefended win.**

---

## Change Log

| Version | Date | Changes |
|---|---|---|
| v1–v3 | 2026-08-26 | Original build plan (`2026-08-26-africa-growth-ml-capstone.md`) |
| **v4** | **2026-08-27** | **Remediation plan. Addresses 23 review findings across 6 phases.** |

---

## READ THIS FIRST — Three things that will change how you work

### 1. The headline result is already known. Do not try to beat it.

The reviewer ran the experiments. Here are **verified** numbers from the committed panel (recompute them yourself in Task 2.1; do not trust this table blindly, but expect to reproduce it):

| Model (fit on train 2000–2017) | Val MAE | Test MAE |
|---|---:|---:|
| Global-mean baseline | 4.038 | **1.896** |
| Persistence baseline | 5.021 | 2.229 |
| Country historical-mean baseline | — | 1.982 |
| Shipped HGB (`max_iter=1000, depth=5`, refit train+val) | 3.906 | **3.540** |
| **Tuned HGB (`depth=2, lr=0.01, l2=1.0`, early stopping, fit train-only)** | 3.887 | **1.821** |

The tuned model reaches test MAE **1.821 vs 1.896** for the baseline — a **+0.075pp** improvement. Paired bootstrap over test residuals (5,000 resamples):

```
95% CI on paired MAE improvement = [-0.047, +0.193]  →  spans zero
```

**The model achieves parity with the global mean. The improvement is not statistically significant.** That is the honest finding, and it is what you will report. Do **not** hunt for a configuration that wins — that is test-set fishing and is exactly how C2 happened. Your job is to make the *process* correct and the *reporting* truthful.

The capstone value proposition shifts accordingly, and this is defensible and mature:
> *"Fourteen WDI indicators observed at year t carry no statistically significant information about year t+1 GDP per capita growth beyond the unconditional mean. This is a substantive finding about the predictability of African macro growth at annual frequency, demonstrated with a leakage-free protocol and a pre-registered decision rule."*

### 2. `data/raw/` is absent from this checkout.

`ls data/raw/` → does not exist. Only `data/processed/model_data.parquet` (1300×18, 52 countries, 2000–2024) is committed.

- **Tasks that only re-model** (Phases 1–3) work from the committed parquet. No download needed.
- **Task 1.4 (add Mauritius + Sudan)** *requires* re-downloading `WDI_CSV.zip`, because those countries have no rows in the committed panel. It has an explicit fallback if download is impossible — read it before starting.

### 3. Definition of done includes a gate that can fail the build.

You will add a **baseline gate** that refuses to write model artifacts unless the selected model beats both baselines *on validation*. If your model fails that gate, **that is a correct outcome** — fix the model or accept and document the null result. Do not weaken or bypass the gate to make the build pass.

---

## AGENTS.md compliance (applies to every task)

| Rule | Enforcement in this plan |
|---|---|
| 1. Verify code | Every task ends with a verification command; paste real output |
| 2. Tests for every `src/*.py` | Maintained; **plus** new `tests/test_finalize_model.py` and `tests/test_app.py` (H5) |
| 3. Docstrings + type hints | All new functions |
| 4. Logging, no `print()` | `print()` stays banned in `src/`, `scripts/`, `app.py`. Notebooks may print |
| 5. No silent failures | Gate raises `SystemExit`; M4 logging bug fixed |
| 6. No leakage | Test set touched **exactly once**, in Task 2.4 |
| 7. Lean notebooks | Both notebooks re-executed, logic stays in `src/` |
| 8. Deliberate visuals | Existing palette retained |
| 9. Minimal dependencies | Only `nbformat`/`nbconvert` (dev extra) added |
| 10. Follow spec | Streamlit only; no new frameworks |
| 11. Smallest change | Prefer editing existing functions over rewriting modules |
| 12. Final status | Each task response: what changed, tests run, result, `VERIFIED`/`NOT VERIFIED` |

---

## Finding → Task traceability

Every review finding maps to at least one task. Use this as your final checklist.

| Finding | Severity | Task(s) |
|---|---|---|
| C1 Model loses to constant; no gate | Critical | 2.2, 2.3, 2.4 |
| C2 Notebook selects on test | Critical | 3.2 |
| C3 Fabricated report statistics | Critical | 4.1, 4.2, 4.3 |
| C4 Claimed tuning does not exist | Critical | 2.1, 4.2 |
| C5 HGB overfit (`early_stopping` inert) | Critical | 2.1, 2.2 |
| C6 COVID refit → −2.07pp bias | High | 2.3, 2.4, 4.2 |
| H1 Importance is noise, mislabelled "direction" | High | 2.5, 3.2, 4.2 |
| H2 `ColumnTransformer` permutes order | High | 1.1 |
| H3 Guardrails on wrong distribution | High | 3.3 |
| H4 Directional accuracy degenerate | High | 1.2, 4.2 |
| H5 `finalize_model.py` + `app.py` untested | High | 5.1, 5.2 |
| M1 README pipeline broken | Medium | 4.4 |
| M2 Notebooks hardcode `C:\dev\...` | Medium | 3.1 |
| M3 MUS/SDN missing; count wrong | Medium | 1.4 |
| M4 Bootstrap logging `TypeError` | Medium | 1.3 |
| M5 `pyproject.toml` packaging broken | Medium | 1.5 |
| M6 No duplicate check | Medium | 1.6 |
| M7 No provenance/versioning | Medium | 2.6 |
| M8 No PDF report | Medium | 4.5 |
| M9 No deploy link / screenshot | Medium | 6.1, 6.2 |
| M10 `.streamlit` CORS + port | Medium | 6.3 |
| M11 `pytest` in runtime reqs | Medium | 1.5 |
| M12 Dead code | Medium | 1.6 |

---

## Phase 0 — Setup

### Task 0.1: Environment and green baseline

- [ ] **Step 1 — Create venv and install**

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

- [ ] **Step 2 — Confirm starting state**

```bash
pytest -q
```

**Acceptance:** `33 passed`. If not, stop and report — the baseline is not what this plan assumes.

- [ ] **Step 3 — Snapshot current artifacts for later comparison**

```bash
mkdir -p .artifacts_pre
cp models/model_metadata.json .artifacts_pre/
cp models/test_predictions.parquet .artifacts_pre/
```

Add `.artifacts_pre/` to `.gitignore`. This is scratch, never committed.

---

## Phase 1 — Correctness fixes (no modelling changes)

> Independent, low-risk fixes. Complete before touching the model so Phase 2 runs on a clean base.

### Task 1.1: Fix `ColumnTransformer` feature-order trap (H2)

**Files:** `src/train.py`, `tests/test_train.py`

`build_ridge_pipeline` emits `[log(C), A, B, D]` when `C` is the log feature — order is permuted, so `zip(feature_names, coef_)` mislabels every coefficient. Task 2.5 needs correct Ridge coefficients, so fix this first.

- [ ] **Step 1 — Add an order-preserving accessor to `src/train.py`**

```python
def get_transformed_feature_names(pipeline: Pipeline, input_features: List[str]) -> List[str]:
    """Return output feature names in the order the final estimator sees them.

    A ColumnTransformer concatenates transformer outputs in declaration order,
    so the log-transformed column is moved to position 0. Mapping coefficients
    with zip(input_features, coef_) is therefore incorrect whenever a
    'log_transform' step is present.

    Args:
        pipeline: Fitted or unfitted Pipeline, optionally containing a
            'log_transform' ColumnTransformer step.
        input_features: Feature names in the order passed to .fit().

    Returns:
        Output feature names aligned positionally with the final estimator's
        coef_ / feature_importances_ array.
    """
    if "log_transform" not in pipeline.named_steps:
        return list(input_features)
    ct = pipeline.named_steps["log_transform"]
    names: List[str] = []
    for name, _transformer, cols in ct.transformers_:
        if name == "remainder":
            continue
        for idx in cols:
            base = input_features[idx]
            names.append(f"{base}_log1p" if name == "log" else base)
    return names
```

- [ ] **Step 2 — Add a regression test to `tests/test_train.py`**

```python
def test_get_transformed_feature_names_reflects_column_permutation():
    """ColumnTransformer moves the log column to position 0; names must follow."""
    from src.train import get_transformed_feature_names
    feats = ["A", "B", "C", "D"]
    pipe = build_ridge_pipeline(alpha=1.0, log_transform_features=["C"],
                                all_feature_names=feats)
    X = pd.DataFrame(np.random.rand(20, 4), columns=feats)
    pipe.fit(X, pd.Series(np.random.rand(20)))
    names = get_transformed_feature_names(pipe, feats)
    assert names == ["C_log1p", "A", "B", "D"], names
    assert len(names) == len(pipe.named_steps["model"].coef_)


def test_get_transformed_feature_names_identity_without_log_step():
    from src.train import get_transformed_feature_names
    feats = ["A", "B"]
    pipe = build_ridge_pipeline(alpha=1.0)
    assert get_transformed_feature_names(pipe, feats) == feats
```

- [ ] **Step 3 — Verify**

```bash
pytest tests/test_train.py -q -k transformed_feature_names
```

**Acceptance:** 2 passed.

---

### Task 1.2: Add honest directional-accuracy metrics (H4)

**Files:** `src/train.py`, `tests/test_train.py`

80.7% of test targets are ≥ 0, so the current metric awards 80.7% to any always-positive constant. Keep the raw metric (comparability) but add the two that carry skill.

- [ ] **Step 1 — Extend `compute_metrics` in `src/train.py`**

Add to the returned dict, leaving existing keys unchanged:

```python
    # H4: raw directional accuracy equals the majority-class rate for any
    # constant-sign predictor. Report skill-aware companions alongside it.
    actual_pos = y_true >= 0
    pred_pos = y_pred >= 0
    majority_rate = max(actual_pos.mean(), 1.0 - actual_pos.mean())

    tpr = (pred_pos[actual_pos]).mean() if actual_pos.any() else np.nan
    tnr = (~pred_pos[~actual_pos]).mean() if (~actual_pos).any() else np.nan
    balanced = np.nanmean([tpr, tnr])

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy": direction_correct,
        "directional_majority_rate": float(majority_rate),
        "directional_skill": float(direction_correct - majority_rate),
        "balanced_directional_accuracy": float(balanced),
    }
```

Update the docstring to state that `directional_accuracy` is **not** a skill measure and must always be quoted next to `directional_majority_rate`.

- [ ] **Step 2 — Test the degeneracy explicitly**

```python
def test_directional_metrics_expose_majority_class_degeneracy():
    """A constant positive predictor must score 0 skill, not high accuracy."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, -1.0])   # 80% positive
    y_pred = np.full(5, 2.0)                         # always positive
    m = compute_metrics(y_true, y_pred)
    assert m["directional_accuracy"] == pytest.approx(0.8)
    assert m["directional_majority_rate"] == pytest.approx(0.8)
    assert m["directional_skill"] == pytest.approx(0.0)
    assert m["balanced_directional_accuracy"] == pytest.approx(0.5)
```

- [ ] **Step 3 — Verify**

```bash
pytest tests/test_train.py -q
```

**Note:** downstream code reads metrics by key, so extra keys are additive. If any test asserts an exact dict, update it.

---

### Task 1.3: Fix bootstrap logging `TypeError` (M4)

**Files:** `src/evaluate.py`, `tests/test_evaluate.py`

- [ ] **Step 1 — Fix line ~95**

```python
    logger.info(
        "Bootstrap CI (%.0f%%): [%.4f, %.4f] (n=%d)",
        confidence * 100, lower, upper, n_bootstrap
    )
```

(`%%.0f%%` consumed no argument, leaving 4 args for 3 placeholders.)

- [ ] **Step 2 — Test that logging emits cleanly**

```python
def test_compute_bootstrap_ci_logs_without_formatting_error(caplog):
    """M4: the CI log line must not raise a logging TypeError."""
    import logging
    a = np.random.RandomState(0).randn(30)
    p = a + 0.1
    with caplog.at_level(logging.INFO):
        compute_bootstrap_ci(a, p, lambda x, y: np.mean(np.abs(x - y)), n_bootstrap=25)
    assert any("Bootstrap CI" in r.getMessage() for r in caplog.records)
```

`record.getMessage()` raises if formatting is broken, so this genuinely guards the bug.

- [ ] **Step 3 — Verify**

```bash
pytest tests/test_evaluate.py -q
python -c "
import logging, numpy as np, sys; sys.path.insert(0,'.')
logging.basicConfig(level=logging.INFO)
from src.evaluate import compute_bootstrap_ci
compute_bootstrap_ci(np.random.randn(40), np.random.randn(40), lambda a,b: np.mean(np.abs(a-b)), 50)
"
```

**Acceptance:** no `--- Logging error ---` on stderr.

---

### Task 1.4: Complete the African country list (M3)

**Files:** `config/indicators.yaml`, `tests/test_config.py`, possibly `data/processed/*`

Config has 53 codes; docs claim 54. **Mauritius (MUS) and Sudan (SDN) are missing entirely** — no warning was ever emitted. ESH (Western Sahara) is present but has no WDI rows.

- [ ] **Step 1 — Add both codes** to `geographic.african_countries` in `config/indicators.yaml`, keeping alphabetical-by-region grouping. Result: **55 entries** (54 UN member states + ESH).

- [ ] **Step 2 — Annotate ESH** directly above its entry:

```yaml
    # ESH (Western Sahara): not a UN member state; retained for completeness.
    # WDI publishes no rows for ESH, so src.data logs it as missing-from-data.
    - ESH
```

- [ ] **Step 3 — Replace the weak assertion in `tests/test_config.py`**

```python
UN_AFRICAN_MEMBER_STATES = {
    "DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COG",
    "COD","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN",
    "GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ",
    "NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN",
    "TZA","TGO","TUN","UGA","ZMB","ZWE",
}


def test_all_un_african_member_states_present():
    """M3: MUS and Sudan were silently absent. Assert full coverage."""
    config = load_config(Path("config/indicators.yaml"))
    missing = UN_AFRICAN_MEMBER_STATES - set(config.african_countries)
    assert not missing, f"Missing UN African member states: {sorted(missing)}"


def test_country_list_has_no_unexpected_entries():
    config = load_config(Path("config/indicators.yaml"))
    extra = set(config.african_countries) - UN_AFRICAN_MEMBER_STATES
    assert extra <= {"ESH"}, f"Unexpected non-member entries: {sorted(extra)}"
```

- [ ] **Step 4 — Regenerate the panel (CONDITIONAL)**

Config alone does not add data. Choose one path:

**Path A — WDI download available (preferred).**
```bash
mkdir -p data/raw && cd data/raw
# Download WDI_CSV.zip from
# https://datatopics.worldbank.org/world-development-indicators/  → Bulk Downloads → CSV
unzip WDI_CSV.zip && cd ../..
python -m src.data
python -m src.features
python -c "
import pandas as pd
d = pd.read_parquet('data/processed/model_data.parquet')
print('countries:', d.iso3.nunique())
assert {'MUS','SDN'} <= set(d.iso3.unique()), 'MUS/SDN still absent'
print('OK — MUS and SDN present')
"
```
Expect ~54 countries and ~1350 rows. **All Phase 2 numbers must then be regenerated from this panel** — treat the reference table in this plan as indicative only, and re-derive every figure.

**Path B — download impossible.**
Do **not** fake it. Keep the config fix (it is correct and prevents recurrence), then:
1. Add to `data/README.md` under a new `## Known Data Limitations` heading:
   > The committed `model_data.parquet` was generated before MUS and SDN were added to the country list and therefore covers 52 countries. Re-running `python -m src.data && python -m src.features` against a fresh `WDI_CSV.zip` will produce the full 54-country panel.
2. Mirror that sentence in report §3 *Data Limitations* and README's data section.
3. Mark `test_all_un_african_member_states_present` as passing (it tests config, not the panel) and add:
```python
@pytest.mark.xfail(reason="Committed panel predates MUS/SDN; see data/README.md")
def test_panel_covers_all_configured_countries():
    import pandas as pd
    panel = pd.read_parquet("data/processed/model_data.parquet")
    cfg = load_config(Path("config/indicators.yaml"))
    expected = set(cfg.african_countries) - {"ESH"}
    assert expected <= set(panel["iso3"].unique())
```

- [ ] **Step 5 — Verify**

```bash
pytest tests/test_config.py -q
```

**State explicitly in your task report which path you took.**

---

### Task 1.5: Fix packaging and dependency hygiene (M5, M11)

**Files:** `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`

`packages.find.where = ["src"]` finds nothing (`src/__init__.py` makes `src` itself the package), so `pip install -e .` exposes no `src.*` modules — which is why the notebooks needed a hardcoded path.

- [ ] **Step 1 — Fix `pyproject.toml`**

Replace the `[tool.setuptools.packages.find]` block with:

```toml
[tool.setuptools]
packages = ["src"]
```

- [ ] **Step 2 — Split dependencies.** Remove `pytest>=7.0` from `requirements.txt` (M11 — never ship a test runner to the Streamlit runtime). Create `requirements-dev.txt`:

```txt
-r requirements.txt
pytest>=7.0
jupyter>=1.0
nbconvert>=7.0
ipykernel>=6.0
```

Keep `[project.optional-dependencies].dev` in `pyproject.toml` in sync.

- [ ] **Step 3 — Verify the editable install now works**

```bash
pip install -e ".[dev]"
cd /tmp && python -c "import src.config; print('src.config importable from', __import__('os').getcwd())" && cd -
```

**Acceptance:** import succeeds from a directory that is not the repo root. This is the precondition for Task 3.1.

---

### Task 1.6: Duplicate detection and dead-code removal (M6, M12)

**Files:** `src/data.py`, `tests/test_data.py`

Spec §7 requires investigating conflicting `(iso3, year)` duplicates rather than silently dropping them. `pivot_table(aggfunc="first")` does exactly what the spec prohibits.

- [ ] **Step 1 — Add `check_duplicates` to `src/data.py`**

```python
def check_duplicates(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """Detect duplicate keys and surface conflicting values.

    Spec section 7 requires that conflicting duplicates be investigated, not
    silently collapsed. Exact duplicates are safe to drop; conflicting ones
    indicate a source-data problem and are logged at WARNING.

    Args:
        df: Long-format frame prior to pivoting.
        key_cols: Columns forming the uniqueness key,
            e.g. ["iso3", "year", "indicator_code"].

    Returns:
        Rows belonging to duplicated keys, empty if none.
    """
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    dups = df[dup_mask]
    if dups.empty:
        logger.info("Duplicate check on %s: none found", key_cols)
        return dups

    exact = df.duplicated(keep=False) & dup_mask
    n_exact = int(exact.sum())
    conflicting = dups[~dups.index.isin(df[exact].index)]
    logger.warning(
        "Duplicate check on %s: %d duplicated rows (%d exact, %d conflicting)",
        key_cols, len(dups), n_exact, len(conflicting),
    )
    if not conflicting.empty:
        logger.warning(
            "Conflicting duplicate keys require investigation:\n%s",
            conflicting.head(20).to_string(),
        )
    return dups
```

- [ ] **Step 2 — Call it in the `__main__` block of `src/data.py`**, after `reshape_wide_to_long` and before `pivot_to_country_year`:

```python
        check_duplicates(long, ["iso3", "year", "indicator_code"])
        panel = pivot_to_country_year(long)
```

- [ ] **Step 3 — Remove dead code (M12).** In `filter_african_countries`, delete the unreachable `aggregate_codes` set (it also contains duplicate literals) — the explicit ISO3 list already excludes aggregates. Replace with a one-line comment:

```python
    # Aggregates (SSF, AFE, AFW, WLD, income groups) cannot appear: the
    # config list contains only sovereign ISO3 codes. See tests/test_data.py.
```

Keep `clean_numeric` — it is a genuine safety net for the raw-CSV path even though `pivot_table` usually pre-coerces.

- [ ] **Step 4 — Tests**

```python
def test_check_duplicates_flags_conflicting_values(caplog):
    import logging
    df = pd.DataFrame({
        "iso3": ["GHA", "GHA", "KEN"],
        "year": [2018, 2018, 2018],
        "indicator_code": ["X", "X", "X"],
        "value": [1.0, 2.0, 3.0],          # conflicting
    })
    with caplog.at_level(logging.WARNING):
        dups = check_duplicates(df, ["iso3", "year", "indicator_code"])
    assert len(dups) == 2
    assert "conflicting" in caplog.text.lower()


def test_check_duplicates_clean_panel_returns_empty():
    df = pd.DataFrame({
        "iso3": ["GHA", "KEN"], "year": [2018, 2018],
        "indicator_code": ["X", "X"], "value": [1.0, 2.0],
    })
    assert check_duplicates(df, ["iso3", "year", "indicator_code"]).empty
```

- [ ] **Step 5 — Verify**

```bash
pytest tests/test_data.py -q
```

---

### Phase 1 gate

```bash
pytest -q
```

**Acceptance:** all green (expect ~41 tests). Commit:

```bash
git add -A
git commit -m "fix: correctness fixes for H2, H4, M3, M4, M5, M6, M11, M12"
```

---

## Phase 2 — Rebuild the modelling protocol (C1, C4, C5, C6, H1, M7)

> **The heart of the remediation.** After Task 2.1, the test set is sealed. It is read **exactly once**, in Task 2.4.

### Task 2.1: Add real expanding-window hyperparameter search (C4, C5)

**Files:** `src/train.py`, `tests/test_train.py`

Spec §9 mandates expanding-window tuning. Report §6 claimed it happened. No such code exists. Build it for real.

- [ ] **Step 1 — Add expanding-window CV to `src/train.py`**

```python
def expanding_window_splits(
    years: pd.Series,
    initial_train_end: int,
    val_window: int = 2,
    final_train_end: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate chronological expanding-window (train_idx, val_idx) pairs.

    Implements the spec section 9 protocol:
        train 2000-2010 -> validate 2011-2012
        train 2000-2012 -> validate 2013-2014
        ...
    All folds live strictly inside the training period; the held-out
    validation and test splits are never touched here.

    Args:
        years: Year value per row, index-aligned with the feature matrix.
        initial_train_end: Last year of the first training fold.
        val_window: Number of years in each validation fold.
        final_train_end: Last year available for folding (defaults to max year).

    Returns:
        List of (train_positions, val_positions) integer-position arrays.
    """
    y = years.reset_index(drop=True)
    last = int(final_train_end if final_train_end is not None else y.max())
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    cut = initial_train_end
    while cut + val_window <= last:
        tr = np.where(y <= cut)[0]
        va = np.where((y > cut) & (y <= cut + val_window))[0]
        if len(tr) and len(va):
            splits.append((tr, va))
        cut += val_window
    logger.info(
        "Expanding-window CV: %d folds (initial_train_end=%d, val_window=%d, last=%d)",
        len(splits), initial_train_end, val_window, last,
    )
    return splits


def search_hyperparameters(
    build_fn: Callable[..., Pipeline],
    param_grid: List[Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    initial_train_end: int,
    val_window: int = 2,
) -> pd.DataFrame:
    """Score a compact parameter grid with expanding-window CV.

    Selection uses mean fold MAE with std as a stability tiebreaker, per
    spec section 11 ("stability across temporal folds").

    Args:
        build_fn: Callable returning an unfitted Pipeline for given params.
        param_grid: Explicit list of parameter dicts (compact grid, not random).
        X: Training-period features only.
        y: Training-period target only.
        years: Year per row, index-aligned with X.
        initial_train_end: Last year of the first fold.
        val_window: Validation years per fold.

    Returns:
        One row per configuration with mean_mae, std_mae and fold count,
        sorted by mean_mae ascending.
    """
    splits = expanding_window_splits(years, initial_train_end, val_window)
    if not splits:
        raise ValueError("Expanding-window CV produced no folds; check year range.")

    rows = []
    for params in param_grid:
        fold_maes = []
        for tr_idx, va_idx in splits:
            model = build_fn(**params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = model.predict(X.iloc[va_idx])
            fold_maes.append(mean_absolute_error(y.iloc[va_idx], pred))
        rows.append({
            **params,
            "mean_mae": float(np.mean(fold_maes)),
            "std_mae": float(np.std(fold_maes)),
            "n_folds": len(fold_maes),
        })
        logger.info("CV %s -> mean MAE %.4f (+/- %.4f)",
                    params, rows[-1]["mean_mae"], rows[-1]["std_mae"])
    return pd.DataFrame(rows).sort_values(["mean_mae", "std_mae"]).reset_index(drop=True)
```

Add `Callable` and `Any` to the `typing` import.

- [ ] **Step 2 — Fix the HGB overfit defaults (C5)**

`early_stopping="auto"` is inert below 10,000 samples (n=905), so all 1000 iterations ran. Change `build_hgb_pipeline` to take explicit early-stopping controls:

```python
def build_hgb_pipeline(
    max_iter: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    random_state: int = 42,
    l2_regularization: float = 1.0,
    early_stopping: bool = True,
    validation_fraction: float = 0.15,
    n_iter_no_change: int = 15,
) -> Pipeline:
```

Pass all of them to `HistGradientBoostingRegressor`. **Docstring must note:** `early_stopping="auto"` only activates above 10k samples, so it is set explicitly here.

- [ ] **Step 3 — Update `config/indicators.yaml`** to match the new defaults and add the grid:

```yaml
model:
  random_state: 42
  ridge_alpha: 1.0
  hgb_max_iter: 200
  hgb_learning_rate: 0.05
  hgb_max_depth: 3
  hgb_l2_regularization: 1.0
  hgb_early_stopping: true
  hgb_validation_fraction: 0.15
  hgb_n_iter_no_change: 15

  # Compact grid for expanding-window CV (spec section 9: no large search).
  cv_initial_train_end: 2010
  cv_val_window: 2
  ridge_alpha_grid: [1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0]
  hgb_grid:
    max_depth: [2, 3]
    learning_rate: [0.01, 0.03, 0.05]
    max_iter: [100, 200]
```

Extend the `Config` dataclass and `load_config` with these fields (defaults preserved so existing tests pass).

- [ ] **Step 4 — Tests**

```python
def test_expanding_window_splits_are_chronological_and_disjoint():
    from src.train import expanding_window_splits
    years = pd.Series(list(range(2000, 2018)) * 3).sort_values().reset_index(drop=True)
    splits = expanding_window_splits(years, initial_train_end=2010, val_window=2)
    assert len(splits) >= 3
    for tr, va in splits:
        assert years.iloc[tr].max() < years.iloc[va].min()   # no future leakage
        assert not set(tr) & set(va)                          # disjoint


def test_search_hyperparameters_returns_ranked_grid():
    from src.train import search_hyperparameters, build_ridge_pipeline
    rng = np.random.RandomState(0)
    n = 180
    years = pd.Series(np.repeat(np.arange(2000, 2018), 10))
    X = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n)})
    y = pd.Series(X["a"] * 2 + rng.randn(n) * 0.5)
    res = search_hyperparameters(
        lambda **p: build_ridge_pipeline(**p),
        [{"alpha": 1.0}, {"alpha": 100.0}],
        X, y, years, initial_train_end=2010,
    )
    assert list(res.columns[:1]) == ["alpha"]
    assert {"mean_mae", "std_mae", "n_folds"} <= set(res.columns)
    assert res["mean_mae"].is_monotonic_increasing
```

- [ ] **Step 5 — Verify**

```bash
pytest tests/test_train.py -q
```

---

### Task 2.2: Add the baseline gate (C1)

**Files:** `src/train.py`, `tests/test_train.py`

The root cause of C1: baselines were computed on test, *after* selection. Nothing could fail the build.

- [ ] **Step 1 — Add the gate to `src/train.py`**

```python
def enforce_baseline_gate(
    candidate_metrics: Dict[str, float],
    baseline_metrics: Dict[str, Dict[str, float]],
    metric: str = "mae",
    lower_is_better: bool = True,
) -> Dict[str, Any]:
    """Fail the build unless the candidate beats every baseline on validation.

    C1: the previous pipeline shipped a model 86 percent worse than the global
    mean because baselines were only computed on test, after selection. This
    gate runs on validation, before any artifact is written.

    Args:
        candidate_metrics: Validation metrics for the selected model.
        baseline_metrics: Mapping of baseline name -> validation metrics.
        metric: Metric key to gate on.
        lower_is_better: True for error metrics.

    Returns:
        Report dict with 'passed' plus per-baseline margins.

    Raises:
        ValueError: If a baseline is missing the gating metric.
    """
    if not baseline_metrics:
        raise ValueError("Baseline gate requires at least one baseline.")

    cand = candidate_metrics[metric]
    results, failures = {}, []
    for name, bm in baseline_metrics.items():
        if metric not in bm:
            raise ValueError(f"Baseline '{name}' missing metric '{metric}'.")
        base = bm[metric]
        beats = cand < base if lower_is_better else cand > base
        margin = (base - cand) if lower_is_better else (cand - base)
        results[name] = {
            "baseline_value": float(base),
            "candidate_value": float(cand),
            "margin": float(margin),
            "relative_margin_pct": float(margin / base * 100) if base else float("nan"),
            "passed": bool(beats),
        }
        if not beats:
            failures.append(f"{name} ({metric}={base:.4f} vs candidate {cand:.4f})")
        logger.info("Baseline gate [%s]: candidate %.4f vs %.4f -> %s",
                    name, cand, base, "PASS" if beats else "FAIL")

    report = {"metric": metric, "passed": not failures,
              "failures": failures, "per_baseline": results}
    if failures:
        logger.error("BASELINE GATE FAILED against: %s", "; ".join(failures))
    else:
        logger.info("Baseline gate PASSED against all %d baselines.", len(results))
    return report
```

- [ ] **Step 2 — Tests**

```python
def test_baseline_gate_fails_when_candidate_worse():
    from src.train import enforce_baseline_gate
    rep = enforce_baseline_gate({"mae": 3.54}, {"global_mean": {"mae": 1.90}})
    assert rep["passed"] is False
    assert "global_mean" in rep["failures"][0]


def test_baseline_gate_passes_when_candidate_better():
    from src.train import enforce_baseline_gate
    rep = enforce_baseline_gate(
        {"mae": 1.80},
        {"global_mean": {"mae": 1.90}, "persistence": {"mae": 2.23}},
    )
    assert rep["passed"] is True
    assert rep["per_baseline"]["global_mean"]["margin"] == pytest.approx(0.10)


def test_baseline_gate_reproduces_the_c1_regression():
    """Regression guard for the exact shipped failure."""
    from src.train import enforce_baseline_gate
    rep = enforce_baseline_gate(
        {"mae": 3.5397},
        {"global_mean": {"mae": 1.8958}, "persistence": {"mae": 2.2287}},
    )
    assert rep["passed"] is False
    assert len(rep["failures"]) == 2
```

- [ ] **Step 3 — Verify**

```bash
pytest tests/test_train.py -q -k baseline_gate
```

**Acceptance:** 3 passed.

---

### Task 2.3: Rewrite `finalize_model.py` around a pre-registered protocol (C1, C6)

**Files:** `scripts/finalize_model.py`

Rewrite `main()` in this exact order. **Do not read `y_test` before Step G.**

- [ ] **Step A — Load panel, build target, coverage-filter on train mask.** *(unchanged)*

- [ ] **Step B — Split; apply the B10 fair-comparison filter.** *(unchanged — keep it, it was correct)*

- [ ] **Step C — Compute baselines on VALIDATION.** This is the fix for C1.

```python
    val_baselines = {
        "global_mean": compute_metrics(
            y_val.values, global_mean_baseline(y_train, len(y_val))),
    }
    pers_val = val.loc[y_val.index, config.target_code].values
    pers_ok = ~np.isnan(pers_val)
    if pers_ok.any():
        val_baselines["persistence"] = compute_metrics(
            y_val.values[pers_ok], pers_val[pers_ok])
    logger.info("Validation baselines: %s",
                {k: round(v["mae"], 4) for k, v in val_baselines.items()})
```

- [ ] **Step D — Expanding-window CV inside the training period only.**

```python
    ridge_cv = search_hyperparameters(
        lambda **p: build_ridge_pipeline(
            log_transform_features=log_features,
            all_feature_names=final_features, **p),
        [{"alpha": a} for a in config.ridge_alpha_grid],
        X_train, y_train, train.loc[y_train.index, "year"],
        config.cv_initial_train_end, config.cv_val_window,
    )
    hgb_param_grid = [
        {"max_depth": d, "learning_rate": lr, "max_iter": mi}
        for d in config.hgb_grid["max_depth"]
        for lr in config.hgb_grid["learning_rate"]
        for mi in config.hgb_grid["max_iter"]
    ]
    hgb_cv = search_hyperparameters(
        lambda **p: build_hgb_pipeline(random_state=config.random_state, **p),
        hgb_param_grid, X_train, y_train,
        train.loc[y_train.index, "year"],
        config.cv_initial_train_end, config.cv_val_window,
    )
    ridge_cv.to_csv("models/cv_results_ridge.csv", index=False)
    hgb_cv.to_csv("models/cv_results_hgb.csv", index=False)
```

- [ ] **Step E — Fit CV-best of each family on train, score on validation, pick the winner by validation MAE.** Log both families' validation metrics.

- [ ] **Step F — Run the gate. Refuse to write artifacts on failure.**

```python
    gate = enforce_baseline_gate(winner_val_metrics, val_baselines, metric="mae")
    if not gate["passed"]:
        logger.error(
            "Refusing to write artifacts: selected model does not beat "
            "validation baselines. Failures: %s", gate["failures"])
        if not args.allow_baseline_failure:
            raise SystemExit(2)
        logger.warning(
            "--allow-baseline-failure set: writing artifacts for a model that "
            "FAILED the baseline gate. This must be disclosed in the report.")
```

Add an `argparse` flag `--allow-baseline-failure` (default `False`). It exists so a documented null result can still be shipped **deliberately**; it must never be used to quietly bypass the gate.

- [ ] **Step G — Refit policy (C6). Pre-registered, not test-driven.**

Validation targets are 2019–2021 and include the COVID crash (target mean **−4.76**); test targets are 2022–2024 (mean **+1.87**). Refitting on train+val injects a regime that does not represent test — this is the mechanism behind the −2.07pp bias.

**Pre-registered decision: `refit_strategy = "train_only"` is PRIMARY.** The rationale is a priori (regime mismatch), not chosen by peeking at test.

```python
    REFIT_STRATEGY = "train_only"   # pre-registered; see report section 4
    if REFIT_STRATEGY == "train_only":
        winner_pipeline.fit(X_train, y_train)
    else:
        winner_pipeline.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
```

Compute the `train_val` variant as a **sensitivity analysis** and store both in metadata under `sensitivity.refit_train_val`. Report the primary in headline tables.

- [ ] **Step H — Touch the test set ONCE.** Predict, compute metrics, compute all three test baselines (global mean, persistence, **country historical mean** — spec §10 optional baseline 3, implemented expanding so it uses no future data).

- [ ] **Step I — Paired significance test vs the best baseline.**

```python
    resid_model = np.abs(y_test.values - y_pred_test)
    resid_base = np.abs(y_test.values - gm_pred)
    diff = resid_base - resid_model            # positive => model better
    rng = np.random.RandomState(config.random_state)
    boot = np.array([diff[rng.randint(0, len(diff), len(diff))].mean()
                     for _ in range(5000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    significance = {
        "paired_mae_improvement_vs_global_mean": float(diff.mean()),
        "ci_lower": float(lo), "ci_upper": float(hi),
        "significant_at_95": bool(lo > 0),
        "n_bootstrap": 5000,
    }
    logger.info("Paired MAE improvement %.4f pp, 95%% CI [%.4f, %.4f], significant=%s",
                diff.mean(), lo, hi, lo > 0)
```

- [ ] **Step J — Write artifacts** (pipeline, predictions, importance, metadata, country metadata) — plus the new `cv_results_*.csv`, `gate`, `significance`, and `sensitivity` blocks.

**Acceptance for this task:**
- `grep -n "y_test" scripts/finalize_model.py` shows no read before Step G.
- Gate runs on validation, before artifact writes.
- Script exits non-zero when the gate fails without the flag.

---

### Task 2.4: Execute the sealed test evaluation (C1, C6)

- [ ] **Step 1 — Run it**

```bash
python scripts/finalize_model.py 2>&1 | tee logs/finalize_$(date +%Y%m%d_%H%M%S).log
```

- [ ] **Step 2 — Record the outcome honestly.** Two possibilities:

**(a) Gate passes and test confirms parity/improvement.** Expected: tuned HGB (`depth=2, lr≈0.01–0.03, l2=1.0`, early stopping, train-only refit) → test MAE ≈ **1.82** vs global mean **1.90**, with the paired CI spanning zero. Report as **parity, not victory**:
> The selected model attains test MAE 1.82 against 1.90 for the global-mean baseline. The paired 95% CI on the improvement is [−0.05, +0.19] and includes zero, so the model is **statistically indistinguishable from the unconditional mean**.

**(b) Gate fails on validation.** Do **not** tune against test to escape. Re-examine the grid once; if it still fails, ship deliberately with `--allow-baseline-failure` and lead the report with the null result.

- [ ] **Step 3 — Verify no test leakage into selection**

```bash
python - <<'PY'
import json, pathlib
m = json.load(open("models/model_metadata.json"))
assert "gate" in m and m["gate"]["metric"] == "mae", "gate block missing"
assert "significance" in m, "significance block missing"
for f in ["models/cv_results_ridge.csv", "models/cv_results_hgb.csv"]:
    assert pathlib.Path(f).exists(), f
print("gate passed:", m["gate"]["passed"])
print("test MAE:", round(m["metrics"]["winner_test"]["mae"], 4))
print("global mean test MAE:", round(m["metrics"]["global_mean_baseline"]["mae"], 4))
print("significant at 95%:", m["significance"]["significant_at_95"])
PY
```

- [ ] **Step 4 — Confirm determinism**

```bash
cp models/model_metadata.json /tmp/meta_a.json
python scripts/finalize_model.py >/dev/null 2>&1
python -c "
import json
a=json.load(open('/tmp/meta_a.json')); b=json.load(open('models/model_metadata.json'))
ka=a['metrics']['winner_test']; kb=b['metrics']['winner_test']
assert abs(ka['mae']-kb['mae'])<1e-9, (ka,kb)
print('deterministic OK')
"
```

---

### Task 2.5: Honest feature interpretation (H1)

**Files:** `scripts/finalize_model.py`, `src/evaluate.py`, `tests/test_evaluate.py`

10 of 14 permutation importances were negative (permuting *improves* score) — the signature of no learned signal. These were published as a ranked table with a **"Direction: Positive/Negative"** column, which is a category error: permutation importance has no sign semantics.

- [ ] **Step 1 — Add a significance-aware importance helper to `src/evaluate.py`**

```python
def compute_permutation_importance_with_ci(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    n_repeats: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance with dispersion and a significance flag.

    H1: importance magnitude carries no directional meaning, and values whose
    spread straddles zero are indistinguishable from noise. Callers must not
    render these as 'positive'/'negative' effects.

    Returns:
        Columns: feature, importance_mean, importance_std, ci_lower, ci_upper,
        is_significant (ci_lower > 0), sorted by importance_mean descending.
    """
    res = sk_permutation_importance(
        pipeline, X, y, n_repeats=n_repeats, random_state=random_state)
    lo = np.percentile(res.importances, 2.5, axis=1)
    hi = np.percentile(res.importances, 97.5, axis=1)
    out = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": res.importances_mean,
        "importance_std": res.importances_std,
        "ci_lower": lo,
        "ci_upper": hi,
        "is_significant": lo > 0,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    logger.info("Permutation importance: %d/%d features significant at 95%%",
                int(out["is_significant"].sum()), len(out))
    return out
```

- [ ] **Step 2 — Compute importance on VALIDATION, not test.** The previous run computed it on `X_test, y_test`, reusing the sealed set for interpretation. Use validation. Persist the full frame to `models/feature_importance.parquet` (new columns included).

- [ ] **Step 3 — Add Ridge coefficients (spec §13).** Direction requires a linear model. Fit the CV-best Ridge on train and extract standardized coefficients **using `get_transformed_feature_names` from Task 1.1** — never `zip(feature_names, coef_)`. Save to `models/ridge_coefficients.parquet` with columns `feature, coefficient, abs_coefficient`.

- [ ] **Step 4 — Test**

```python
def test_permutation_importance_ci_flags_noise_features():
    from src.evaluate import compute_permutation_importance_with_ci
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    rng = np.random.RandomState(0)
    n = 200
    X = pd.DataFrame({"signal": rng.randn(n), "noise": rng.randn(n)})
    y = pd.Series(X["signal"] * 3 + rng.randn(n) * 0.2)
    pipe = Pipeline([("i", SimpleImputer()), ("m", Ridge())]).fit(X, y)
    out = compute_permutation_importance_with_ci(
        pipe, X, y, ["signal", "noise"], n_repeats=20)
    assert out.iloc[0]["feature"] == "signal"
    assert bool(out.set_index("feature").loc["signal", "is_significant"]) is True
    assert bool(out.set_index("feature").loc["noise", "is_significant"]) is False
```

- [ ] **Step 5 — Verify**

```bash
pytest tests/test_evaluate.py -q
```

---

### Task 2.6: Provenance and versioning in metadata (M7)

**Files:** `src/train.py` (`write_model_metadata`), `scripts/finalize_model.py`

- [ ] **Step 1 — Extend `write_model_metadata`** to record:

```python
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit_or_none(),
        "library_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "data_provenance": {
            "panel_path": "data/processed/model_data.parquet",
            "panel_sha256": _sha256(Path("data/processed/model_data.parquet")),
            "panel_rows": int(n_rows),
            "n_countries": int(n_countries),
            "year_min": int(year_min),
            "year_max": int(year_max),
            "wdi_vintage": wdi_vintage,   # from data/README.md or CLI arg
        },
        "split_sizes": {"train": int(n_train), "val": int(n_val), "test": int(n_test)},
        "split_target_years": {
            "train": [train_year_min + 1, train_year_max + 1],
            "val":   [val_year_min + 1, val_year_max + 1],
            "test":  [test_year_min + 1, test_year_max + 1],
        },
        "refit_strategy": REFIT_STRATEGY,
        "gate": gate,
        "significance": significance,
        "sensitivity": sensitivity,
```

Implement `_sha256` and `_git_commit_or_none` as small private helpers (the latter returns `None` if `git` is unavailable — never raise).

**`split_target_years` is essential**: it makes explicit that "test 2021+" means *feature* years 2021–2023 and *target* years 2022–2024, the ambiguity that hid C6.

- [ ] **Step 2 — Test**

```python
def test_write_model_metadata_includes_provenance(tmp_path):
    from src.train import write_model_metadata
    p = tmp_path / "meta.json"
    write_model_metadata(
        path=p, feature_names=["a"], target_code="X", train_end=2017,
        val_end=2020, metrics={}, model_type="Ridge", random_state=42,
    )
    meta = json.loads(p.read_text())
    for key in ("created_utc", "library_versions", "split_target_years"):
        assert key in meta, key
```

- [ ] **Step 3 — Verify**

```bash
pytest tests/test_train.py -q
python scripts/finalize_model.py >/dev/null 2>&1
python -c "
import json; m=json.load(open('models/model_metadata.json'))
print(json.dumps({k:m[k] for k in ['created_utc','refit_strategy','split_target_years','split_sizes']}, indent=2))
"
```

---

### Phase 2 gate

```bash
pytest -q && python scripts/finalize_model.py
git add -A && git commit -m "feat: leakage-free selection protocol with baseline gate, real CV, provenance"
```

---

## Phase 3 — Notebooks and app (C2, H3, M2)

### Task 3.1: Make notebooks portable (M2)

**Files:** both notebooks

`PROJECT_ROOT = Path(r'C:\dev\africa-growth-ml')` makes them unrunnable anywhere else; outputs also leak `C:\Users\ingex\...`. Task 1.5 makes `src` properly installable, so the hack is no longer needed.

- [ ] **Step 1 — Replace the root cell in both notebooks**

```python
import os
from pathlib import Path

# Portable root resolution: env var > git root > parent of notebooks/
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "")) if os.environ.get("PROJECT_ROOT") else None
if PROJECT_ROOT is None:
    here = Path.cwd()
    PROJECT_ROOT = next(
        (p for p in [here, *here.parents] if (p / "pyproject.toml").exists()),
        here.parent,
    )
assert (PROJECT_ROOT / "config" / "indicators.yaml").exists(), \
    f"Could not resolve project root (got {PROJECT_ROOT})"

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

- [ ] **Step 2 — Replace every remaining absolute path** with `PROJECT_ROOT / ...`.

- [ ] **Step 3 — Fix `matplotlib` usage.** Drop `matplotlib.use('Agg')` from notebook 02 (it caused `UserWarning: FigureCanvasAgg is non-interactive` on every `plt.show()`); use `%matplotlib inline`.

- [ ] **Step 4 — Re-execute from a clean kernel**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_profiling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_evaluation.ipynb
```

- [ ] **Step 5 — Verify no leaked paths and no empty cells**

```bash
grep -l 'C:\\\\dev\|C:\\\\Users\|ingex' notebooks/*.ipynb && echo "FAIL: paths remain" || echo "OK: no hardcoded paths"
python - <<'PY'
import json, glob
for f in sorted(glob.glob("notebooks/*.ipynb")):
    nb = json.load(open(f))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    empty = [c for c in code if not c.get("outputs")]
    errs = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    print(f"{f}: {len(code)} code cells, {len(empty)} without output, {len(errs)} errors")
    assert not empty and not errs, f
print("all notebooks executed cleanly")
PY
```

---

### Task 3.2: Fix test-set selection in notebook 02 (C2, H1)

**Files:** `notebooks/02_model_evaluation.ipynb`

Cell 13 selects the winner on **test** MAE and prints `Winner: Ridge` while HGB is deployed. Every downstream output describes a model that is not in production.

- [ ] **Step 1 — Delete the notebook's own selection logic.** The notebook must **not** choose a winner. Replace cell 13's logic with a load of the deployed artifact:

```python
import joblib, json
metadata = json.load(open(PROJECT_ROOT / "models" / "model_metadata.json"))
pipeline = joblib.load(PROJECT_ROOT / "models" / "growth_model.joblib")

winner_name = metadata["model_type"]
print(f"Deployed model (selected on validation in scripts/finalize_model.py): {winner_name}")
print(f"Selection gate passed: {metadata['gate']['passed']}")
```

- [ ] **Step 2 — Restructure sections** so validation and test are clearly separated:
  - §2 **Validation** — baselines, expanding-window CV table (load `models/cv_results_*.csv`), family comparison, gate outcome.
  - §3 **Sealed test** — load `models/test_predictions.parquet` (the deployed model's predictions). Do **not** refit anything.
  - §4 Actual-vs-predicted, residuals — from the loaded parquet.
  - §5 Importance — load `models/feature_importance.parquet`; plot only `is_significant` rows, and show the rest greyed or in a separate "not distinguishable from noise" table. Add Ridge coefficients from `models/ridge_coefficients.parquet` for direction.
  - §6 Bootstrap CIs + the paired significance test from `metadata["significance"]`.
  - §7 Error analysis — worst errors, by-country, by-year.
  - §8 COVID placement — must now state the **target-year** framing and the refit decision.

- [ ] **Step 3 — Add an explicit consistency assertion cell** (this is the guard that makes C2 impossible to repeat):

```python
# Guard: notebook numbers must match the deployed artifact exactly.
import numpy as np, pandas as pd
test_preds = pd.read_parquet(PROJECT_ROOT / "models" / "test_predictions.parquet")
nb_mae = float(np.mean(np.abs(test_preds["actual"] - test_preds["predicted"])))
meta_mae = float(metadata["metrics"]["winner_test"]["mae"])
assert abs(nb_mae - meta_mae) < 1e-6, f"Notebook {nb_mae} != metadata {meta_mae}"
print(f"Consistency OK: test MAE {nb_mae:.4f} matches model_metadata.json")
```

- [ ] **Step 4 — Re-execute and verify**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_evaluation.ipynb
python - <<'PY'
import json
nb = json.load(open("notebooks/02_model_evaluation.ipynb"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
assert "hgb_test_metrics[\"mae\"] <= ridge_test_metrics" not in src, "test-set selection still present"
assert "Consistency OK" in json.dumps(nb), "consistency assertion did not run"
print("C2 remediated")
PY
```

---

### Task 3.3: Fix scenario guardrails and app consistency (H3)

**Files:** `app.py`, `tests/test_app.py`

Spec §14 requires **training-data** bounds. The app passes the full panel (2000–2024), so inflation warns at 92.05 instead of 49.51 — a user can set 80% inflation and get no warning.

- [ ] **Step 1 — Add a training-window helper**

```python
@st.cache_data
def get_training_data(data: pd.DataFrame, train_end: int) -> pd.DataFrame:
    """Training-period rows only, for guardrail calibration.

    H3: spec section 14 requires observed TRAINING minimum/maximum. Using the
    full panel silently widens the safe band (inflation P99 92.05 vs 49.51).
    """
    return data[data["year"] <= train_end]
```

- [ ] **Step 2 — Repoint both call sites** (`app.py` ~795–796) to pass `get_training_data(processed_data, metadata["train_end"])` into `get_feature_range` and `get_feature_percentiles`. Also update `check_extrapolation_warning`.

- [ ] **Step 3 — Clamp the slider default (plan v3 item B9, never implemented).** If the observed baseline value falls outside the training range, clamp and inform:

```python
            clamped = float(np.clip(default_val, data_min, data_max))
            if abs(clamped - default_val) > 1e-9:
                st.caption(
                    f"Observed value {default_val:.1f} is outside the training "
                    f"range [{data_min:.1f}, {data_max:.1f}]; slider clamped."
                )
            default_val = clamped
```

- [ ] **Step 4 — Update slider help text** to say the band is the **training** P1–P99, and add a caption noting guardrails reflect the training distribution (2000–{train_end}).

- [ ] **Step 5 — Replace the misleading contribution table.** The current "Approx. Contribution = importance × change" multiplies a *permutation* importance by a raw unit change — dimensionally meaningless, and now doubly so given H1. Replace with a true one-at-a-time model delta:

```python
    for feat, new_val in scenario_changes.items():
        probe = baseline_input.copy()
        probe[feat] = new_val
        delta = float(pipeline.predict(probe)[0] - baseline_pred)
        # column: "Individual effect (pp)" = delta
```

Caption: *"Each row re-runs the model changing only that indicator. Individual effects need not sum to the total because the model is non-linear."*

- [ ] **Step 6 — Surface the honest headline on Model Performance.** Read `metadata["significance"]` and render, e.g.:

> Test MAE 1.82 vs 1.90 for the global-mean baseline. Paired 95% CI [−0.05, +0.19] includes zero: **the model is not statistically distinguishable from predicting the mean.**

Also display `directional_majority_rate` next to `directional_accuracy` wherever the latter appears (H4).

- [ ] **Step 7 — Verify**

```bash
streamlit run app.py --server.headless true --server.port 8501 &
sleep 12 && curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8501/ && kill %1
```

**Acceptance:** HTTP 200, plus the Task 5.2 tests pass.

---

## Phase 4 — Documentation truth (C3, C4, M1, M8)

> **Root cause of C3: numbers were typed by hand.** The structural fix is to generate them. After Task 4.1, no metric may be hand-written in any document.

### Task 4.1: Build the report-asset generator (C3)

**Files:** `scripts/build_report_assets.py` (new), `tests/test_report_assets.py` (new)

- [ ] **Step 1 — Create `scripts/build_report_assets.py`**

It must load only committed artifacts (`model_metadata.json`, `test_predictions.parquet`, `feature_importance.parquet`, `ridge_coefficients.parquet`, `model_data.parquet`, `cv_results_*.csv`) and emit:

1. `reports/generated/metrics.json` — every number any document may cite.
2. `reports/generated/*.md` — ready-to-paste Markdown tables:
   - `table_model_comparison.md` (baselines + both families, val and test)
   - `table_feature_importance.md` (with CI and significance flag)
   - `table_ridge_coefficients.md`
   - `table_yearly_metrics.md`
   - `table_worst_errors.md` (top 10, real rows)
   - `table_eda_summary.md` (min/median/max/coverage per feature, computed)
   - `table_correlations.md` (computed pairwise)
   - `table_cv_results.md`
3. `reports/generated/figures/*.png` — actual-vs-predicted, residuals, importance, correlation heatmap.

Requirements: module docstring, type hints, `logging` only, no `print()`, deterministic, `if __name__ == "__main__": main()`.

- [ ] **Step 2 — Add a doc-consistency test** (`tests/test_report_assets.py`) that fails if any document contains a metric contradicting `metrics.json`:

```python
"""Guards against C3: hand-written numbers drifting from artifacts."""
import json, re
from pathlib import Path
import pytest

DOCS = ["README.md", "reports/capstone_report.md", "presentation/slides_outline.md"]


@pytest.fixture(scope="module")
def metrics():
    p = Path("reports/generated/metrics.json")
    if not p.exists():
        pytest.skip("Run scripts/build_report_assets.py first")
    return json.loads(p.read_text())


def test_documents_quote_correct_test_mae(metrics):
    """Any 'MAE' figure quoted for the deployed model must match metadata."""
    true_mae = round(metrics["winner_test"]["mae"], 2)
    baseline_mae = round(metrics["global_mean_baseline"]["mae"], 2)
    allowed = {f"{true_mae:.2f}", f"{baseline_mae:.2f}",
               f"{round(metrics['persistence_baseline']['mae'], 2):.2f}"}
    pattern = re.compile(r"MAE[^0-9\-]{0,20}(\d+\.\d{2})")
    for doc in DOCS:
        text = Path(doc).read_text(encoding="utf-8")
        for found in pattern.findall(text):
            assert found in allowed, (
                f"{doc} quotes MAE {found}; artifacts allow {sorted(allowed)}")


def test_no_fabricated_worst_error_countries(metrics):
    """C3: report claimed Libya +35pp etc. Worst-error names must be real."""
    real = {r["country_name"] for r in metrics["worst_errors"][:10]}
    report = Path("reports/capstone_report.md").read_text(encoding="utf-8")
    m = re.search(r"### Worst Errors.*?(?=\n##)", report, re.S)
    if not m:
        pytest.skip("No worst-errors section")
    for claimed in re.findall(r"\*\*?([A-Z][a-zA-Z ]+?)\s+20\d{2}", m.group(0)):
        assert claimed.strip() in real, f"'{claimed}' is not in the real top-10"
```

- [ ] **Step 3 — Run and verify**

```bash
python scripts/build_report_assets.py
ls reports/generated/ reports/generated/figures/
python -c "import json; print(json.dumps(json.load(open('reports/generated/metrics.json')), indent=2)[:1500])"
```

---

### Task 4.2: Rewrite the report from generated assets (C3, C4, C6, H1, H4)

**Files:** `reports/capstone_report.md`

Rewrite section by section, pasting from `reports/generated/`. **Every fabricated claim listed in the review must be gone.**

- [ ] **§3 Dataset** — real coverage from `table_eda_summary.md`. Delete *"Worst coverage: FDI, Domestic Credit (~14%)"* (actual: 96.5% / 87.3%). State country count from metadata; add the MUS/SDN note if Task 1.4 Path B.

- [ ] **§4 Methodology** — add a **Split definition** subsection with the target-year table from `split_target_years`:

  | Split | Feature years | Target years | n |
  |---|---|---|---|
  | Train | 2000–2017 | 2001–2018 | … |
  | Val | 2018–2020 | **2019–2021** | … |
  | Test | 2021–2023 | **2022–2024** | … |

  Add **Refit policy (pre-registered)** explaining the COVID regime-mismatch rationale for `train_only`, and cite the `train_val` sensitivity result.

- [ ] **§5 EDA** — replace every statistic with generated values. Specifically correct: growth range (−49.1 to +91.8, *not* −60 to +35), electricity median (42.7, *not* ~55), internet median (7.1, *not* ~15), inflation max (557.2, *not* 400), all four correlations, and the 2000–2010 average growth (2.12, *not* ~4).

- [ ] **§6 Model Development** — describe the **real** expanding-window CV from Task 2.1 and paste `table_cv_results.md`. If Task 1.4 Path B was taken, say so. **Delete the previous fabricated grid paragraph entirely** (C4).

- [ ] **§7 Evaluation** — paste `table_model_comparison.md`, `table_yearly_metrics.md`, `table_worst_errors.md`. Add a **Statistical significance** subsection with the paired CI. Replace *"Mean residual ≈ 0 (unbiased)"* with the computed value and its interpretation.

- [ ] **§8 Interpretation** — rebuild around Ridge coefficients for direction and permutation importance **with CIs** for magnitude. Remove the "Direction: Positive/Negative" column from the permutation table (H1). State plainly how many features are distinguishable from noise. Delete the unemployment/electricity just-so stories unless the feature is significant.

- [ ] **§12 Conclusion** — lead with the honest headline. Suggested framing:

> The strongest configuration reaches test MAE X.XX against Y.YY for the global-mean baseline, with a paired 95% CI of [L, U]. Because that interval includes zero, we conclude that the 14 WDI indicators evaluated here carry **no statistically significant information** about next-year GDP per capita growth beyond the unconditional mean. This is a substantive negative result: it suggests annual-frequency, country-level WDI aggregates are too coarse and too slow-moving to forecast short-run growth, and that meaningful gains likely require higher-frequency or structural data.

- [ ] **Add §13 Threats to Validity** (new): temporal generalization only; COVID regime break; n=150 test observations; multiple-comparison exposure from the CV grid; median imputation; 52-of-54 country coverage if applicable.

- [ ] **Verify**

```bash
pytest tests/test_report_assets.py -q
```

---

### Task 4.3: Update the presentation outline (C3, H4)

**Files:** `presentation/slides_outline.md`

- [ ] Update the metrics table from `table_model_comparison.md`.
- [ ] Slide 5: add the paired CI and state parity explicitly.
- [ ] Remove *"Directional accuracy 52.7% (better than random)"* — random here is the 80.7% majority rate, so this is backwards (H4). Replace with directional **skill**.
- [ ] Slide 8 "Technical Rigor": remove the B-fix table (internal plan IDs mean nothing to an examiner). Replace with: leakage-free protocol, baseline gate, expanding-window CV, pre-registered refit, significance testing.
- [ ] Slide 9: lead with the null result as a *finding*, not an apology.
- [ ] Remove the fabricated demo script numbers (slides ~232–235) or mark them clearly as illustrative.

---

### Task 4.4: Fix README (M1, C3)

**Files:** `README.md`

- [ ] **Fix the broken pipeline section (M1).** `python -m src.train` and `python -m src.evaluate` are no-ops that exit 0; `scripts/finalize_model.py` is never mentioned. Replace with:

```bash
pip install -r requirements.txt
pip install -e ".[dev]"          # dev extras for notebooks/tests

# 1. Download WDI_CSV.zip -> data/raw/ (see data/README.md)
# 2. Build the country-year panel
python -m src.data
python -m src.features

# 3. Select, gate, and finalize the model (writes models/*)
python scripts/finalize_model.py

# 4. Regenerate report assets from the artifacts
python scripts/build_report_assets.py

# 5. Launch the dashboard
streamlit run app.py
```

Either add `__main__` blocks to `src/train.py` / `src/evaluate.py` **or** stop documenting them. Preferred: stop documenting them — `finalize_model.py` is the real entry point.

- [ ] **Replace the performance table** with generated values; add the significance sentence.
- [ ] **Rewrite the feature-importance list** — remove the "strongest positive predictor" framing; mark non-significant entries (H1).
- [ ] **Remove the internal B-fix list** ("B3 FIX", "B4 FIX", …) from the Key Features section — meaningless to an external reader.
- [ ] **Fix the deployment checklist** — `✅ No local absolute paths` was false; re-verify each box after Phase 3.
- [ ] **Add** the deployment URL (Task 6.1), a dashboard screenshot (Task 6.2), and a "Reproducibility" section naming `random_state=42`, the provenance block, and the determinism check.

---

### Task 4.5: Produce the PDF (M8)

**Files:** `reports/capstone_report.pdf`

AnalystLab deliverable #1 is explicitly a PDF.

- [ ] **Step 1 — Convert** (try in order):

```bash
pandoc reports/capstone_report.md -o reports/capstone_report.pdf \
  --pdf-engine=xelatex --toc -V geometry:margin=1in -V fontsize=11pt
# fallback
pandoc reports/capstone_report.md -o reports/capstone_report.html --self-contained --toc
# then print to PDF from a browser
```

- [ ] **Step 2 — Confirm** figures from `reports/generated/figures/` are embedded and referenced in-text.
- [ ] **Step 3 — Commit the PDF** (keep it under ~5 MB; it is a required deliverable, so it *is* tracked).
- [ ] **Verify**

```bash
ls -la reports/capstone_report.pdf && python -c "
print(open('reports/capstone_report.pdf','rb').read(5)[:5])"   # expect b'%PDF-'
```

---

## Phase 5 — Test the untested (H5)

### Task 5.1: Tests for `scripts/finalize_model.py`

**Files:** `tests/test_finalize_model.py` (new)

195 lines containing all selection and artifact logic, with zero tests. **No existing test would have caught C1, C2, or C5.**

- [ ] Make `main()` importable and parameterizable — refactor to `main(config_path=..., panel_path=..., output_dir=..., allow_baseline_failure=False)` with the CLI as a thin wrapper.

- [ ] Write tests using a **synthetic panel fixture** (~20 countries × 25 years) written to `tmp_path`:

```python
def test_finalize_writes_all_expected_artifacts(tmp_path, synthetic_panel): ...
def test_finalize_exits_nonzero_when_gate_fails(tmp_path, unlearnable_panel):
    """Pure-noise target must fail the gate and refuse to write artifacts."""
    with pytest.raises(SystemExit) as e:
        main(panel_path=unlearnable_panel, output_dir=tmp_path)
    assert e.value.code == 2
    assert not (tmp_path / "growth_model.joblib").exists()

def test_finalize_allow_flag_writes_but_records_failure(tmp_path, unlearnable_panel):
    main(panel_path=unlearnable_panel, output_dir=tmp_path, allow_baseline_failure=True)
    meta = json.loads((tmp_path / "model_metadata.json").read_text())
    assert meta["gate"]["passed"] is False

def test_metadata_feature_contract_matches_pipeline(tmp_path, synthetic_panel):
    """Deployed pipeline must accept exactly metadata['feature_names']."""

def test_test_predictions_rows_match_test_split(tmp_path, synthetic_panel): ...
def test_selection_uses_validation_not_test(tmp_path, synthetic_panel):
    """Adversarial: corrupt test targets; selection must be unchanged."""
```

That last test is the direct regression guard for C2 — corrupt the test split and assert the chosen `model_type` and validation metrics are byte-identical.

- [ ] **Verify:** `pytest tests/test_finalize_model.py -q`

---

### Task 5.2: Tests for `app.py`

**Files:** `tests/test_app.py` (new)

913 lines, zero tests. Test the pure helpers — no Streamlit runtime needed (guard the import with `pytest.importorskip("streamlit")`).

- [ ] Cover:

```python
def test_prepare_scenario_input_column_order_matches_contract(): ...
def test_prepare_scenario_input_preserves_nan_for_imputer(): ...
def test_get_feature_percentiles_uses_training_window_only():
    """H3: full-panel P99 for inflation is 92.05; training-only is 49.51."""
def test_check_extrapolation_warning_fires_outside_training_band(): ...
def test_get_scenario_features_returns_three_to_five(): ...
def test_get_feature_display_name_falls_back_to_code(): ...
def test_safe_get_country_data_returns_none_when_missing(): ...
def test_end_to_end_prediction_from_committed_artifacts():
    """Load real artifacts, build one row, predict a finite number."""
```

- [ ] **Verify:** `pytest tests/test_app.py -q`

---

### Task 5.3: Full-suite gate

```bash
pytest -q --tb=short
```

**Acceptance:** all pass, **≥ 60 tests**, none skipped except documented `xfail` from Task 1.4 Path B.

```bash
git add -A && git commit -m "test: cover finalize_model and app; regression guards for C1, C2, C5, H3"
```

---

## Phase 6 — Deployment evidence (M9, M10)

### Task 6.1: Deploy and capture the URL

- [ ] Push the branch; deploy on Streamlit Cloud from `app.py`, Python 3.11.
- [ ] Confirm on the live URL: all 4 pages render; country selector works; a scenario prediction returns; an out-of-band slider value fires the extrapolation warning; the causal disclaimer is visible.
- [ ] Put the URL in README (top badge + Deployment section) and on presentation slide 10.

**If deployment is impossible**, say so explicitly in README rather than implying it is live, and keep the local run instructions as the verified path.

### Task 6.2: Screenshots (spec §20 item 13)

- [ ] Capture `docs/screenshots/{overview,performance,scenario}.png`; embed at least one in README.
- [ ] Keep each under ~500 KB.

### Task 6.3: Fix `.streamlit/config.toml` (M10)

- [ ] Remove `enableCORS = false` (triggers a security warning at boot) and `port = 8501` (conflicts with Cloud's managed port). Keep `[theme]`, `headless`, `gatherUsageStats`.

```toml
[theme]
primaryColor = "#1B4F72"
backgroundColor = "#F8F9FA"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#2C3E50"
font = "sans serif"

[server]
headless = true

[browser]
gatherUsageStats = false
```

- [ ] **Verify:** restart the app; boot log shows no CORS warning.

---

## Final acceptance checklist

Run top to bottom. Every line must pass before declaring completion.

### Automated

```bash
# 1. Full suite
pytest -q                                  # >= 60 passed, 0 failed

# 2. Pipeline reproduces deterministically
python scripts/finalize_model.py
python scripts/build_report_assets.py

# 3. Gate + significance recorded
python -c "
import json; m=json.load(open('models/model_metadata.json'))
assert 'gate' in m and 'significance' in m and 'created_utc' in m
print('gate passed:', m['gate']['passed'])
print('test MAE:', round(m['metrics']['winner_test']['mae'],4),
      '| GM:', round(m['metrics']['global_mean_baseline']['mae'],4))
print('significant:', m['significance']['significant_at_95'])"

# 4. No hardcoded paths anywhere
! grep -rn 'C:\\\\dev\|C:\\\\Users\|/Users/' --include=*.py --include=*.ipynb --include=*.md . \
  --exclude-dir=.git --exclude-dir=.opencode && echo "OK"

# 5. No print() in production code
! grep -rn 'print(' --include=*.py src/ scripts/ app.py && echo "OK"

# 6. Docs agree with artifacts
pytest tests/test_report_assets.py -q

# 7. Notebooks executed, no errors, no empty cells
python - <<'PY'
import json, glob
for f in sorted(glob.glob("notebooks/*.ipynb")):
    nb=json.load(open(f)); code=[c for c in nb["cells"] if c["cell_type"]=="code"]
    assert all(c.get("outputs") for c in code), f
    assert not [o for c in code for o in c.get("outputs",[]) if o.get("output_type")=="error"], f
print("notebooks OK")
PY

# 8. Deployment artifacts not ignored
git check-ignore models/growth_model.joblib data/processed/model_data.parquet && echo "FAIL" || echo "OK"

# 9. App boots
streamlit run app.py --server.headless true --server.port 8501 &
sleep 12 && curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8501/ && kill %1
```

### Manual — findings closed

- [ ] **C1** Gate runs on validation before artifacts; failure exits non-zero
- [ ] **C2** Notebook loads deployed artifact; no test-set selection; consistency assertion runs
- [ ] **C3** Every document number traced to `reports/generated/metrics.json`
- [ ] **C4** Expanding-window CV implemented; `cv_results_*.csv` committed
- [ ] **C5** `early_stopping=True` explicit; `n_iter_ < max_iter`; train/val gap sane
- [ ] **C6** Refit pre-registered + documented; sensitivity reported; target years explicit
- [ ] **H1** Importance has CIs; direction from Ridge only; noise features marked
- [ ] **H2** `get_transformed_feature_names` used everywhere coefficients are mapped
- [ ] **H3** Guardrails use training window; verified inflation warns above ~49.5
- [ ] **H4** Majority rate + skill reported wherever directional accuracy appears
- [ ] **H5** `finalize_model.py` and `app.py` tested
- [ ] **M1** README steps run clean on a fresh clone
- [ ] **M2** Notebooks portable
- [ ] **M3** 54 UN states in config; panel state documented
- [ ] **M4** No logging errors
- [ ] **M5** `pip install -e .` exposes `src.*`
- [ ] **M6** Duplicate check runs and logs
- [ ] **M7** Provenance in metadata
- [ ] **M8** PDF committed
- [ ] **M9** URL + screenshot (or documented absence)
- [ ] **M10** `.streamlit` cleaned
- [ ] **M11** `pytest` out of runtime reqs
- [ ] **M12** Dead code removed

### Manual — brief and spec

- [ ] AnalystLab Steps 1–8 all satisfiable from the repo
- [ ] Deliverables: PDF report, source code, GitHub repo + README + data link, presentation outline
- [ ] Spec §9 temporal validation + expanding-window tuning — **now real**
- [ ] Spec §10 three baselines (global mean, persistence, country historical mean)
- [ ] Spec §12 metrics incl. bootstrap CIs
- [ ] Spec §13 Ridge coefficients **and** permutation importance
- [ ] Spec §14 guardrails on training bounds
- [ ] Spec §17 single pipeline artifact + metadata
- [ ] Spec §20 all 17 README items

---

## Commit sequence

```
fix:   correctness fixes for H2, H4, M3, M4, M5, M6, M11, M12          (Phase 1)
feat:  leakage-free selection protocol with baseline gate and real CV  (Phase 2)
feat:  provenance and versioning in model metadata                     (Phase 2)
fix:   portable notebooks; remove test-set selection from evaluation   (Phase 3)
fix:   scenario guardrails use training distribution                   (Phase 3)
feat:  generate report assets from artifacts                           (Phase 4)
docs:  rewrite report, README, slides from generated numbers           (Phase 4)
test:  cover finalize_model and app; regression guards                 (Phase 5)
chore: deployment config, screenshots, live URL                        (Phase 6)
```

---

## Effort estimate

| Phase | Tasks | Estimate |
|---|---|---|
| 0 Setup | 1 | 15 min |
| 1 Correctness | 6 | 2.5 h |
| 2 Modelling protocol | 6 | 4 h |
| 3 Notebooks & app | 3 | 2.5 h |
| 4 Documentation | 5 | 3.5 h |
| 5 Tests | 3 | 2.5 h |
| 6 Deployment | 3 | 1 h |
| **Total** | **27** | **~16 h** |

Phase 2 is the highest-risk block: if the gate fails, budget an extra hour to interrogate the grid — but resist the urge to tune against test.

---

## Handoff notes for the implementing agent

1. **Read `.opencode/reviews/2026-08-27-capstone-review-v1.md` in full first.** This plan assumes that context.
2. **Work the phases in order.** Phase 2 depends on Task 1.1 (feature-name mapping) and Task 1.2 (metrics). Phase 4 depends on Phase 2 artifacts.
3. **Never hand-type a metric into a document.** If a number is not in `reports/generated/metrics.json`, add it to the generator.
4. **The test set is read exactly once**, in Task 2.4. If you find yourself computing test metrics to make a decision, stop — that is C2 recurring.
5. **A failing baseline gate is a valid, reportable outcome.** Do not weaken the gate; document the result.
6. **Per AGENTS.md rule 12**, end every task response with: what changed, tests run, verification result, and `VERIFIED` / `NOT VERIFIED`.
7. **Flag any deviation from this plan explicitly** with reasoning, rather than silently improvising — silent deviation is how the previous cycle produced a report describing work that had not been done.
