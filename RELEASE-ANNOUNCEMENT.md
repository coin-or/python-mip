# python-mip 1.17 Released 🎉

We are pleased to announce the release of **python-mip 1.17**, the first new release since
1.15.0. This version brings a new solver backend, major infrastructure improvements, and several
bug fixes.

---

## What's New

### HiGHS is now a supported solver

python-mip 1.17 ships with full support for [HiGHS](https://highs.dev) — a high-performance,
open-source LP/MIP solver with an MIT licence. HiGHS joins CBC and Gurobi as a first-class
backend. It supports LP and MIP solve, warm-starting for LP re-solves, and the full
python-mip constraint/variable API.

To use HiGHS, install the optional dependency:

```
pip install mip[highs]
```

This work was led primarily by **Robert Schwarz**, with contributions from **Túlio Toffolo**,
**Adeel Khan**, **Bernard Zweers**, and **Miguel Hisojo**. The integration spanned many months
of careful incremental work — thank you all!

### CBC binary distribution: a new era with `cbcbox`

One of the longest-standing pain points in python-mip has been shipping up-to-date CBC binaries.
Historically, pre-built `.so`/`.dylib`/`.dll` files lived directly in the python-mip repository,
which meant updating CBC required a full python-mip release and manually building binaries for
each platform.

**With 1.17, we have completely decoupled CBC binary distribution** into a new companion package,
[cbcbox](https://pypi.org/project/cbcbox/). `cbcbox` ships pre-built CBC wheels for:

- Linux x86\_64 and aarch64
- macOS x86\_64 and arm64 (Apple Silicon, native — no Rosetta!)
- Windows x64

`cbcbox` is installed automatically as a dependency of python-mip. Future CBC upgrades can now
be shipped by releasing a new `cbcbox` version — completely independently of python-mip.

A big thank you to **Túlio Toffolo** for also building the first macOS ARM64 CBC binary for the
transition period, and for co-developing the `cbcbox` tooling.

### Automated releases via GitHub Actions

Starting with this release, publishing a new version of python-mip to PyPI is as simple as
pushing a version tag:

```
git tag v1.17 && git push --tags
```

A GitHub Actions workflow using OIDC Trusted Publisher takes care of building and uploading
to PyPI automatically, with no API tokens to manage.

### Python 3.10–3.13 + PyPy 3.11 support; minimum raised to 3.10

python-mip now officially supports Python 3.10, 3.11, 3.12, 3.13 and PyPy 3.11, tested across
Linux (x86\_64 and arm64), macOS (Apple Silicon), and Windows. Python 3.8 and 3.9 have reached
end-of-life and are no longer supported.

---

## Bug Fixes

- **CBC re-solve correctness**: Calling `optimize()` multiple times on the same model could
  return stale results due to a behavioural change in newer CBC. Fixed.
- **Empty `LinExpr` in constraints** handled correctly (thanks **Sebastian Heger**, #237).
- **Windows DLL loading** fixed for Python 3.8+ (`os.add_dll_directory` now used).

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

PyPI: https://pypi.org/project/mip/1.17/
