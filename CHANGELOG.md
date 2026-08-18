# Changelog — python-mip 2.0

> Changes since **1.17.1**

---

## Performance

- `LinExpr.add_var` and `xsum` hot paths optimised (single dict operation instead of two;
  bypass dispatch overhead for the common `Var`-only case), cutting 11–21% off constraint-heavy
  model build times.
- HiGHS backend gained a CFFI C-array cache (matching CBC's bulk-add architecture) that
  accumulates pending columns/rows and flushes via `Highs_addCols`/`Highs_addRows`, plus
  name→index dicts to avoid unnecessary flushes on variable/constraint lookups.
- Anonymous variables no longer pay the `str.format`/`str.encode` cost of auto-generated
  `var(N)` names.

A new reproducible benchmark suite (`benchmarks/`, documented in `docs/bench.rst`) was added to
measure and track these improvements — comparing python-mip/CBC, python-mip/HiGHS, and
python-mip/Gurobi model-build speed across CPython 3.14 and PyPy 3.11, against raw `gurobipy`
and `highspy` (high-level and batch-numpy APIs), on four problems (n-Queens, TSP, RCPSP, and the
new Capacitated Facility Location Problem). Highlights:
- python-mip/Gurobi on PyPy is 2–4× faster than on CPython, depending on model structure, and
  remains the only way to drive Gurobi from PyPy at all (`gurobipy` has no PyPy wheel).
- For sparse, variable-heavy models python-mip/Gurobi is competitive with or faster than raw
  `gurobipy` even on CPython; for dense-constraint models `gurobipy`'s own loop is faster on
  CPython, but PyPy closes much of that gap.
- Against `highspy`'s own recommended vectorized numpy-array API, python-mip/HiGHS is
  competitive with or faster than it on sparser/irregular problems (n-Queens, TSP), while the
  vectorized API pulls ahead on dense, rectangular problems (e.g. CFLP).

---

## New Features

### `Model.add_vars()` batch variable creation

`Model.add_vars(n, ...)` / `VarList.add_vars(n, ...)` create `n` variables in a single call,
avoiding per-call dispatch overhead compared to `n` calls to `add_var()`.

### New LP solve controls: `TRUNCATED` status, `max_iter`, `lp_method`, `lp_preprocess`

- `OptimizationStatus.TRUNCATED` for LP solves stopped early by a time or iteration limit
  (populates primal/dual solution and objective bound whenever feasible, instead of reporting
  a misleading `OPTIMAL`/`INFEASIBLE`).
- `Model.max_iter` caps simplex/barrier iterations for LP solves, wired through CBC, HiGHS and
  Gurobi.
- `Model.optimize(lp_method=...)` selects dual simplex, primal simplex or IPM/barrier — added
  to HiGHS and Gurobi (CBC already had it).
- New CBC-specific `LP_Method.RACING` and `LP_Method.RECOMMEND` values: `RACING` runs several
  LP configurations (dual simplex, primal/Idiot, primal/Sprint) in parallel threads and keeps
  the first to reach optimality (needs `Model.threads >= 2`; safely degrades to `RECOMMEND`
  with a log message if fewer threads are available); `RECOMMEND` uses CBC's ML-based
  per-instance LP method selection. Both are ignored (fall back to their own default) by
  HiGHS and Gurobi. `LP_Method.AUTO` (the default) already delegates to `RECOMMEND` when
  sequential and `RACING` when 2+ threads are set, so no behaviour changes for existing code —
  these are purely new, explicit opt-in values.
- `Model.optimize(relax=True, lp_preprocess=True)` (CBC-specific) enables CBC's
  `INT_PARAM_LP_FAST_PREPROCESS` knapsack bound-tightening before an LP relaxation solve.

---

## Infrastructure & Distribution

- `cbcbox` bumped to `>=2.935`, improving binary reliability across platforms.
- HiGHS backend migrated to `highspy`; added `TRUNCATED` status and `max_iter`/`lp_method`
  (dual/primal simplex, IPM/barrier) support, also ported to the Gurobi backend.
- Python 3.14 added to the CI test matrix (alongside 3.10–3.13 and PyPy 3.11).

---

## Bug Fixes

- CBC string parameters migrated to the `INT_PARAM` API; fixed `verbose` reset on `clear()`.
- Removed an obsolete objective-sense save/restore workaround around `Cbc_reset()`.
- Stale `highspy`→`highsbox` error message corrected.

---

## Previous release: python-mip 1.17 / 1.17.1

## New Features

### HiGHS Solver Support
python-mip now ships with full support for the [HiGHS](https://highs.dev) open-source solver as a
first-class backend (alongside CBC and Gurobi). HiGHS is a high-performance solver for LP and MIP
problems with a permissive MIT licence.

Key capabilities added:
- Full LP and MIP solve via HiGHS C API (through `highsbox`)
- Warm-start (basis handoff) for LP re-solves
- `relax=True` support in `optimize()`
- Variable and constraint inspection/modification
- Correct handling of `UNBOUNDED` vs `INFEASIBLE` status
- Reduced memory footprint and improved file read/write consistency

HiGHS is installed as an optional dependency: `pip install mip[highs]`.

### macOS Apple Silicon (M1/M2/M3) Native Support
CBC now runs natively on Apple Silicon via a pre-built ARM64 binary, replacing the previous
Rosetta 2 x86_64 fallback.

---

## Infrastructure & Distribution

### CBC Binaries via `cbcbox`
The bundled CBC shared libraries (`.so`, `.dylib`, `.dll`) have been **removed from the
python-mip source tree**. CBC binaries are now distributed through the
[cbcbox](https://pypi.org/project/cbcbox/) PyPI package, which provides pre-built wheels for:

- Linux x86\_64 and aarch64 (ARM64)
- macOS x86\_64 and arm64
- Windows x64

`cbcbox` is a dedicated package whose sole job is to ship up-to-date CBC binaries for all
major platforms. This decoupling means future CBC upgrades are released without touching
python-mip itself. The minimum required version is `cbcbox>=2.902`.

### Automated PyPI Publishing
A new GitHub Actions workflow (`.github/workflows/publish.yml`) automatically publishes to
PyPI whenever a `v*` tag is pushed. It uses OIDC Trusted Publisher authentication — no API
tokens to rotate.

### Modernised CI Matrix
| Platform | OS |
|---|---|
| Linux x86\_64 | ubuntu-24.04 |
| Linux aarch64 | ubuntu-24.04-arm *(new)* |
| macOS ARM64 | macos-15 *(new)* |
| Windows x64 | windows-2025 *(new)* |

Python versions tested: **3.10, 3.11, 3.12, 3.13, PyPy 3.11**.

---

## Bug Fixes

- **CBC re-solve correctness**: A bug introduced by newer CBC versions caused stale solution
  data to be returned when `optimize()` was called multiple times on the same model. Fixed by
  calling `Cbc_reset()` before each `Cbc_solve()`, with objective sense saved and restored
  around the reset.
- **`isfile` import missing in `SolverCbc.read()`**: `os.path.isfile` was used but not
  imported, causing a `NameError` when loading a model from a file.
- **Windows DLL loading**: On Python 3.8+, Windows ignores `PATH` when resolving DLL
  dependencies. Fixed by calling `os.add_dll_directory()` on the cbcbox `bin/` directory.
- **Empty `LinExpr` in constraints**: Constraints containing an empty linear expression were
  not handled correctly. Fixed by Sebastian Heger (#237).

---

## Breaking Changes / Compatibility

- **Minimum Python version raised to 3.10.** Python 3.8 and 3.9 have reached end-of-life and
  are no longer tested or supported.
- Bundled CBC libraries removed — `cbcbox` is now a required dependency (installed
  automatically via pip).
- `gurobipy` version constraint relaxed to `>=10` (no upper bound).
- `cffi` version constraint relaxed to `>=1.15` (no upper bound).
- `highsbox` version constraint relaxed to `>=1.10.0` (no upper bound).

---

## Acknowledgements

This release was a team effort. Thank you to everyone who contributed:

- **Robert Schwarz** — HiGHS interface: initial implementation (PR #332) and extensive
  improvements (PR #418), including objective setter fix, option types, test coverage and
  `highsbox` migration. Co-authored with **Bernard Zweers** and **Miguel Hisojo**.
- **Túlio Toffolo** — macOS Apple Silicon support, HiGHS testing infrastructure, CI
  modernisation, and many quality-of-life fixes.
- **Sebastian Heger** — Bug fix for constraints with empty linear expressions (#237).
- **Dominik Peters** — Removed upper limit on supported Python versions (#408).
- **Adeel Khan** — HiGHS `_core` library support.
- **Haroldo Santos** — cbcbox integration, CBC bug fixes, CI/CD automation, and release management.
