# python-mip 2.0 Released 🎉

We are pleased to announce **python-mip 2.0**, a major milestone release. Since 1.17.1, CBC
and HiGHS binary distribution has been hardened across platforms (`cbcbox 2.935`), model-building
performance has improved further, and new LP
solve controls (`TRUNCATED` status, `max_iter`, `lp_method`, CBC's `lp_preprocess`) have been
added across all three backends. To measure and validate these improvements, we've also
published a comprehensive, reproducible benchmark suite comparing python-mip's model-creation
speed against raw `gurobipy` and `highspy`, including a fair, apples-to-apples comparison
against `highspy`'s own recommended vectorized API and a look at what's uniquely possible with
**Gurobi on PyPy**.

---

## What's New in 2.0

### Model-build performance improvements

- `LinExpr.add_var` and `xsum` hot paths were optimised (single dict operation instead of two,
  bypassing dispatch overhead for the common `Var`-only case), cutting **11–21%** off
  constraint-heavy model build times.
- New `Model.add_vars(n, ...)` / `VarList.add_vars(n, ...)` batch API creates `n` variables in
  one call — one bounds check and one `list.extend()` instead of `n` separate `add_var()`
  calls.
- HiGHS gained a CFFI C-array cache (matching CBC's existing bulk-add architecture) that
  accumulates pending columns/rows and flushes via `Highs_addCols`/`Highs_addRows`, plus
  name→index dicts to avoid unnecessary flushes on variable/constraint lookups.

To measure and validate these improvements — and to answer a question we kept getting asked
("how does this compare to using `gurobipy`/`highspy` directly?") — this release adds a
reproducible benchmark suite (`benchmarks/`, documented in [`docs/bench.rst`](docs/bench.rst))
covering four problems (n-Queens, TSP single-commodity flow, RCPSP, and the new **Capacitated
Facility Location Problem**) across CPython 3.14 and PyPy 3.11.

**Gurobi on PyPy**: python-mip's Gurobi backend talks to the Gurobi C API directly via CFFI, not
through `gurobipy` — which has quietly meant, for many releases now, that python-mip is the
*only* way to drive Gurobi from PyPy (`gurobipy` has no PyPy wheel and never has). The benchmarks
confirm this is a real, usable advantage, not just a theoretical one:
- On **PyPy 3.11**, python-mip/Gurobi model creation is **2–4× faster** than on CPython,
  depending on model structure.
- For sparse, variable-heavy models (e.g. n-Queens, 1.44M binary variables), python-mip/Gurobi
  is competitive with or faster than raw `gurobipy` even on CPython at most sizes tested, and
  keeps building models that make `gurobipy` time out.
- For dense-constraint applied models (e.g. capacitated facility location), `gurobipy`'s own
  loop-based API is faster on CPython — but python-mip/Gurobi on PyPy still closes much of that
  gap, building models in the time `gurobipy` can't even finish on CPython.

**HiGHS vs. `highspy`**: we compared against `highspy`'s own recommended vectorized numpy-array
API (`addBinaries`/`addVariables`/`addConstrs`), not a naive loop, for the fairest comparison:
- On dense, rectangular structures where numpy's vectorization shines (e.g. our new CFLP
  benchmark), `highspy`'s vectorized API is the fastest non-batch backend of all.
- On sparser or irregular structures (e.g. n-Queens, TSP), python-mip/HiGHS is competitive with
  or faster than the vectorized `highspy` API, and clearly faster once problem size grows past
  what fits comfortably in a dense array.
- `highspy`'s batch numpy API (hand-built CSR matrix, no incremental modelling API) remains the
  fastest option everywhere, but under PyPy both `highspy` APIs get *slower* than on CPython
  since PyPy doesn't JIT-compile numpy — while python-mip's backends keep scaling smoothly.

See the appendix in `docs/bench.rst` for a discussion of code clarity trade-offs between
python-mip's incremental modelling style and `highspy`'s vectorized numpy API.

### New LP solve controls: `TRUNCATED` status, `max_iter`, `lp_method`, `lp_preprocess`

- `OptimizationStatus.TRUNCATED` is now returned when an LP solve is stopped early by a time or
  iteration limit, instead of being reported as `OPTIMAL`/`INFEASIBLE`. Primal/dual solution and
  objective bound are populated whenever the corresponding feasibility conditions hold.
- `Model.max_iter` lets you cap the number of simplex/barrier iterations for LP solves; wired
  through CBC, HiGHS and Gurobi.
- `Model.optimize(lp_method=...)` selects dual simplex, primal simplex, or IPM/barrier — added
  to HiGHS and Gurobi (CBC already supported it).
- New CBC-specific `LP_Method.RACING`: races several LP configurations (dual simplex,
  primal/Idiot crash, primal/Sprint) across parallel threads and keeps whichever reaches
  optimality first. Requires `Model.threads >= 2`; with fewer threads it safely falls back to
  `LP_Method.RECOMMEND` (CBC's ML-based per-instance method selector) with a log message —
  never an error. Both are no-ops on HiGHS/Gurobi. The default `LP_Method.AUTO` already
  delegates to `RECOMMEND` sequentially and `RACING` once 2+ threads are configured, so this is
  a purely additive, opt-in change — verified locally with matching objective values across
  `AUTO`/`DUAL`/`PRIMAL`/`RACING`/`RECOMMEND` on the same LP.
- `Model.optimize(relax=True, lp_preprocess=True)` (CBC-specific) enables CBC's fast
  knapsack-based bound-tightening preprocessing before an LP relaxation solve.

### CBC and HiGHS binary updates

- `cbcbox` bumped to `>=2.935`, improving binary reliability across platforms.
- HiGHS backend migrated to `highspy`.
- Python 3.14 added to the CI test matrix (alongside 3.10–3.13 and PyPy 3.11).

---

## What's New in 1.17.1

### CBC crash fix: Gomory Mixed-Integer cuts (`CutType.GMI`)

A crash was discovered when using `CutType.GMI` with recent CBC versions. The root cause was
a bug in `scaleCutIntegral` — a function shared by `OsiCuts` and `CglGomory` — that modified
coefficient arrays **in place** before asserting integrality, causing an `abort()` when
borderline floating-point rounding left a value just outside the 1e-9 tolerance. The fix
(pre-check all values before applying any modification) was committed directly to the upstream
COIN-OR repositories [coin-or/Osi](https://github.com/coin-or/Osi) and
[coin-or/Cgl](https://github.com/coin-or/Cgl) and is included in `cbcbox 2.910`.

### Updated CBC binaries via `cbcbox 2.910`

python-mip 1.17.1 requires `cbcbox>=2.910`, which ships binaries built from the latest
COIN-OR master (post-fix). The new wheels include:

- The `scaleCutIntegral` crash fix in both Osi and Cgl
- Updated `CutType` C enum (`CT_LaGomory` removed; entries renumbered) correctly reflected
  in python-mip's `ffi.cdef`
- Performance improvements from recent COIN-OR master commits

### cbcbox: faster CBC, simpler releases

`cbcbox` is now the sole distribution channel for CBC binaries. This decoupling means future
CBC improvements — algorithm enhancements, bug fixes, new COIN-OR master commits — can reach
users with a `cbcbox` release alone, without touching python-mip at all. The release cycle
for CBC upgrades is now:

1. Push fix to upstream COIN-OR (coin-or/Cbc, coin-or/Cgl, coin-or/Osi, …)
2. Bump `cbcbox` version and push — CI builds all platforms automatically
3. Bump `cbcbox>=X.Y` in python-mip `pyproject.toml` and push a tag

On x86\_64 (Linux, macOS, Windows), `cbcbox` ships **two** complete solver stacks per wheel:

| Variant | OpenBLAS kernel | Description |
|---|---|---|
| `generic` | `DYNAMIC_ARCH` runtime dispatch | Compatible with any x86\_64 CPU |
| `avx2` | `HASWELL` 256-bit AVX2/FMA | Optimised for Haswell (2013+) and newer |

The best variant is selected automatically at import time. The AVX2 build delivers measurable
speedups on modern hardware thanks to wider SIMD in the dense linear algebra kernels used by
Clp's simplex solver.

---

## What's New in 1.17 (first release since 1.15.0)

### HiGHS is now a supported solver

python-mip 1.17 ships with full support for [HiGHS](https://highs.dev) — a high-performance,
open-source LP/MIP solver with an MIT licence. HiGHS joins CBC and Gurobi as a first-class
backend. It supports LP and MIP solve, warm-starting for LP re-solves, and the full
python-mip constraint/variable API.

HiGHS binaries are distributed via `highsbox`, installed automatically as an optional
dependency:

```
pip install mip[highs]
```

This work was led primarily by **Robert Schwarz**, with contributions from **Túlio Toffolo**,
**Adeel Khan**, **Bernard Zweers**, and **Miguel Hisojo**. The integration spanned many months
of careful incremental work — thank you all!

### CBC binary distribution via `cbcbox`

Historically, pre-built CBC binaries lived directly in the python-mip repository, requiring
a full python-mip release for every CBC update and manual cross-platform builds.

**With 1.17, CBC binary distribution is fully decoupled** into
[cbcbox](https://pypi.org/project/cbcbox/), a dedicated companion package with pre-built
wheels for:

- Linux x86\_64 and aarch64
- macOS x86\_64 and arm64 (Apple Silicon, native — no Rosetta!)
- Windows x64

`cbcbox` is installed automatically. The same architecture applies to HiGHS via `highsbox`.

### Automated releases via GitHub Actions

Publishing a new version of python-mip to PyPI is now as simple as:

```
git tag v1.17.1 && git push --tags
```

A GitHub Actions workflow using OIDC Trusted Publisher handles building and uploading to
PyPI with no API tokens to manage.

### Python 3.10–3.13 + PyPy 3.11; minimum raised to 3.10

python-mip now officially supports Python 3.10, 3.11, 3.12, 3.13 and PyPy 3.11, tested
across Linux (x86\_64 and arm64), macOS (Apple Silicon), and Windows. Python 3.8 and 3.9
have reached end-of-life and are no longer supported.

---

## Bug Fixes (cumulative 1.17 + 1.17.1)

- **GMI cut crash** (`CutType.GMI`): `scaleCutIntegral` assert in Osi/Cgl. Fixed upstream.
- **CBC re-solve correctness**: `optimize()` called multiple times could return stale results.
  Fixed by calling `Cbc_reset()` before each solve.
- **Empty `LinExpr` in constraints** handled correctly (thanks **Sebastian Heger**, #237).
- **Windows DLL loading** fixed for Python 3.8+ (`os.add_dll_directory` now used).
- **Stale HiGHS error message** corrected to reference `highsbox` instead of `highspy`.

---

## Upgrading

```
pip install --upgrade mip
```

For HiGHS support:

```
pip install --upgrade "mip[highs]"
```

---

## Contributors

Thank you to everyone who contributed code, bug reports, and reviews since 1.15:

**Robert Schwarz** · **Túlio Toffolo** · **Sebastian Heger** · **Dominik Peters** ·
**Adeel Khan** · **Bernard Zweers** · **Miguel Hisojo** · **Abdullah Hasani** ·
**Haroldo Santos**

---

Full changelog: https://github.com/coin-or/python-mip/blob/master/CHANGELOG.md

PyPI: https://pypi.org/project/mip/2.0.0/
