# python-mip 1.17.1 Released 🎉

We are pleased to announce the release of **python-mip 1.17.1**, the first publicly available
release in quite some time. This is a patch release on top of 1.17, which itself brought a new
solver backend, major infrastructure improvements, and several bug fixes. 1.17.1 delivers
important stability fixes for CBC and updated binaries via `cbcbox 2.910`.

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
**Adeel Khan** · **Bernard Zweers** · **Miguel Hisojo** · **Haroldo Santos**

---

Full changelog: https://github.com/coin-or/python-mip/blob/master/CHANGELOG.md

PyPI: https://pypi.org/project/mip/1.17.1/
