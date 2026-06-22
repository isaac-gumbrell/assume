<!--
SPDX-FileCopyrightText: ASSUME Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Failing test: `test_load_srmc_congestion_from_db` — Diagnosis & Fix

## Symptom

```
FAILED tests/test_loader_csv.py::test_load_srmc_congestion_from_db
AssertionError: assert {'congestion_L1', 'congestion_L2'} == {'L1_congestion_signal', 'L2_congestion_signal'}
  Extra items in the left set:  'congestion_L1', 'congestion_L2'
  Extra items in the right set: 'L1_congestion_signal', 'L2_congestion_signal'
```

The test expects the columns returned by `load_srmc_congestion_from_db` to be named
`{line_id}_congestion_signal` (e.g. `L1_congestion_signal`). The function actually
returns `congestion_{line_id}` (e.g. `congestion_L1`).

This failure is **unrelated to the staggered-training mask work** — it pre-exists
those changes (none of them touch `load_srmc_congestion_from_db`).

## Root cause

**The test is stale; the production code is correct.**

`load_srmc_congestion_from_db` (`assume/scenario/loader_csv.py`) pivots `grid_flows`
rows and renames the columns to the `congestion_` **prefix** convention:

```python
wide.columns = [f"congestion_{col}" for col in wide.columns]
```

This is the convention required by the **only runtime consumer** of these columns,
the preprocess algorithm `congestion_signal_lines_load_from_df`
(`assume/common/forecast_algorithms.py`), which recovers the line id by stripping a
literal `congestion_` prefix:

```python
prefix = "congestion_"
cols = {
    col[len(prefix):]: col
    for col in forecast_df.columns
    if col.startswith(prefix)
}
```

The pipeline ties them together in `load_scenario_data`: the SRMC result is joined
straight into `forecasts_df`, which the preprocess step then scans for
`congestion_*` columns:

```python
srmc_df = load_srmc_congestion_from_db(db_uri, srmc_sim_id, index)
...
forecasts_df = forecasts_df.join(srmc_df, how="outer")
```

If the columns were named `L1_congestion_signal`, the `startswith("congestion_")`
filter would never match and **every SRMC congestion signal would be silently
dropped** at runtime. So `congestion_{line_id}` is the correct, load-bearing name.

### Why the test drifted

| What | Commit | Date | Naming it uses |
|---|---|---|---|
| `tests/test_loader_csv.py` last touched | `814e5cff` | 2026-05-28 | `{line_id}_congestion_signal` (old) |
| `load_srmc_congestion_from_db` fixed to prefix convention | `1f314f15` *"fix(congestion-forecast): use prefix convention and tighten defaults"* | 2026-06-03 | `congestion_{line_id}` (new) |

Commit `1f314f15` switched the function (and its consumer) to the `congestion_`
prefix convention but **did not update this test**, leaving it asserting the
superseded naming.

## Fix

Update the three column-name references in
`tests/test_loader_csv.py::test_load_srmc_congestion_from_db` from
`{line}_congestion_signal` to `congestion_{line}`:

```python
# before
assert set(result.columns) == {"L1_congestion_signal", "L2_congestion_signal"}
assert pytest.approx(result["L1_congestion_signal"].tolist()) == [0.2, 0.4, 0.6]
assert pytest.approx(result["L2_congestion_signal"].tolist()) == [0.1, 0.1, 0.1]

# after
assert set(result.columns) == {"congestion_L1", "congestion_L2"}
assert pytest.approx(result["congestion_L1"].tolist()) == [0.2, 0.4, 0.6]
assert pytest.approx(result["congestion_L2"].tolist()) == [0.1, 0.1, 0.1]
```

The other assertions (`len(result) == 3`, `result.index.equals(index)`, the
numeric values) are correct and stay unchanged.

## Recommended companion clean-ups (stale docs, no behaviour change)

The same outdated `{line_id}_congestion_signal` naming survives in a few
doc/comment strings and should be corrected to `congestion_{line_id}` to avoid
re-introducing the confusion:

1. `assume/scenario/loader_csv.py` — `load_srmc_congestion_from_db` docstring
   **Returns** section says
   `DataFrame with columns {line_id}_congestion_signal ...`. This contradicts the
   function body (which says `congestion_{line_id}`). Change it to
   `congestion_{line_id}`.
2. `assume/scenario/loader_csv.py` — the inline comment in `load_scenario_data`
   ("merge them as `{line_id}_congestion_signal` columns into forecasts_df").
   Change to `congestion_{line_id}`.
3. `assume/common/forecaster.py` (≈ line 306) — the `initialize` docstring lists
   `congestion_*` *or* `*_congestion_signal` as accepted columns, but only the
   `congestion_*` prefix is actually implemented by
   `congestion_signal_lines_load_from_df`. Drop the `*_congestion_signal`
   alternative (or implement it) so the docs match the code.

## Validation

After editing the test:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_loader_csv.py::test_load_srmc_congestion_from_db -q --no-header -p no:cacheprovider
```

Expect `1 passed`. Then re-run the loader module to confirm no regressions:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_loader_csv.py -q --no-header -p no:cacheprovider
```
