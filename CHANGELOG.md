# Changelog — python-mip 1.17

> Changes since **1.15.0** (1.16 was an unreleased RC)

---

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
