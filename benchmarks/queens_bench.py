"""N-Queens model-building and solving benchmark.

Constructs a binary integer program for the n-queens problem and measures
model creation time across different solver backends and Python interpreters.
The largest default instance (n=300) has 90,000 binary variables and roughly
4,000 constraints.

Benchmarks:
  - python-mip with CBC       (CFFI C-array column/row cache)
  - python-mip with HiGHS     (CFFI C-array column/row cache)
  - python-mip with Gurobi    (if available)
  - highspy native high-level API (addVariable/addConstr expressions)
  - highspy native batch API  (addVars/addRows with numpy CSR arrays)

Usage:
    python queens_bench.py [sizes]          # e.g. 100 200 300 400 500
    python queens_bench.py --build-only     # skip solve timing
    python queens_bench.py --solve-only     # skip build-only timing
"""

import sys
import time
import platform

SIZES = [100, 200, 300, 400, 500]

# ── helpers ──────────────────────────────────────────────────────────────────

def hline():
    print("-" * 72)

def row(label, *vals):
    print(f"  {label:<28}", end="")
    for v in vals:
        print(f"  {v:>10}", end="")
    print()


def _queens_constraints(n):
    """Return row/col/diag constraint lists as (indices, coeffs) tuples.
    Used by batch APIs."""
    import numpy as np
    rows_idx, rows_val = [], []
    # row constraints: sum_j x[i*n+j] == 1
    for i in range(n):
        idx = np.arange(i * n, (i + 1) * n, dtype=np.int32)
        rows_idx.append(idx)
        rows_val.append(np.ones(n))
    # col constraints: sum_i x[i*n+j] == 1
    for j in range(n):
        idx = np.array([i * n + j for i in range(n)], dtype=np.int32)
        rows_idx.append(idx)
        rows_val.append(np.ones(len(idx)))
    # diagonal \: i - j == k
    for k in range(2 - n, n - 1):
        idx = np.array([i * n + (i - k) for i in range(n) if 0 <= i - k < n], dtype=np.int32)
        if len(idx) >= 2:
            rows_idx.append(idx)
            rows_val.append(np.ones(len(idx)))
    # diagonal /: i + j == k
    for k in range(2, 2 * n - 1):
        idx = np.array([i * n + (k - i) for i in range(n) if 0 <= k - i < n], dtype=np.int32)
        if len(idx) >= 2:
            rows_idx.append(idx)
            rows_val.append(np.ones(len(idx)))
    return rows_idx, rows_val


# ── python-mip benchmark ─────────────────────────────────────────────────────

def bench_pmip(n, solver_name, max_solve_sec=10.0):
    import mip
    from mip import Model, xsum, BINARY

    t0 = time.perf_counter()
    m = Model(solver_name=solver_name)
    m.verbose = 0

    x = [[m.add_var(var_type=BINARY) for j in range(n)] for i in range(n)]

    for i in range(n):
        m += xsum(x[i][j] for j in range(n)) == 1
    for j in range(n):
        m += xsum(x[i][j] for i in range(n)) == 1
    for k in range(2 - n, n - 1):
        cells = [x[i][i - k] for i in range(n) if 0 <= i - k < n]
        if len(cells) >= 2:
            m += xsum(cells) <= 1
    for k in range(2, 2 * n - 1):
        cells = [x[i][k - i] for i in range(n) if 0 <= k - i < n]
        if len(cells) >= 2:
            m += xsum(cells) <= 1

    t_build = time.perf_counter() - t0

    m.max_seconds = max_solve_sec
    t1 = time.perf_counter()
    status = m.optimize()
    t_solve = time.perf_counter() - t1

    return t_build, t_solve, str(status).split(".")[-1]


# ── highspy high-level benchmark ─────────────────────────────────────────────

def bench_highspy_hl(n, max_solve_sec=10.0):
    """highspy with high-level addVariable/addConstr expressions."""
    import highspy

    t0 = time.perf_counter()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)

    inf = highspy.kHighsInf
    x = [[h.addVariable(lb=0, ub=1) for j in range(n)] for i in range(n)]
    # mark integer
    kInteger = highspy.HighsVarType.kInteger
    for i in range(n):
        for j in range(n):
            h.changeColIntegrality(i * n + j, kInteger)

    for i in range(n):
        h.addConstr(sum(x[i][j] for j in range(n)) == 1)
    for j in range(n):
        h.addConstr(sum(x[i][j] for i in range(n)) == 1)
    for k in range(2 - n, n - 1):
        cells = [x[i][i - k] for i in range(n) if 0 <= i - k < n]
        if len(cells) >= 2:
            h.addConstr(sum(cells) <= 1)
    for k in range(2, 2 * n - 1):
        cells = [x[i][k - i] for i in range(n) if 0 <= k - i < n]
        if len(cells) >= 2:
            h.addConstr(sum(cells) <= 1)

    t_build = time.perf_counter() - t0

    h.setOptionValue("time_limit", max_solve_sec)
    t1 = time.perf_counter()
    h.run()
    t_solve = time.perf_counter() - t1

    ms = h.getModelStatus()
    return t_build, t_solve, h.modelStatusToString(ms)


# ── highspy batch (numpy) benchmark ──────────────────────────────────────────

def bench_highspy_batch(n, max_solve_sec=10.0):
    """highspy with batch numpy addVars/addRows API."""
    import highspy
    import numpy as np

    t0 = time.perf_counter()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)

    n2 = n * n
    h.addVars(n2, np.zeros(n2), np.ones(n2))
    # mark all as integer: changeColsIntegrality(num_set, set_indices, types)
    all_cols = np.arange(n2, dtype=np.int32)
    integrality = np.full(n2, highspy.HighsVarType.kInteger, dtype=np.uint8)
    h.changeColsIntegrality(n2, all_cols, integrality)

    rows_idx, rows_val = _queens_constraints(n)
    num_rows = len(rows_idx)

    # Build CSR representation
    nz_counts = [len(r) for r in rows_idx]
    total_nz = sum(nz_counts)
    starts = np.zeros(num_rows + 1, dtype=np.int32)
    for i, c in enumerate(nz_counts):
        starts[i + 1] = starts[i] + c
    all_idx = np.concatenate(rows_idx).astype(np.int32)
    all_val = np.concatenate(rows_val)

    # Determine bounds for each constraint type
    lower = np.empty(num_rows)
    upper = np.empty(num_rows)
    # First 2n are equality (== 1), rest are <= 1
    lower[: 2 * n] = 1.0
    upper[: 2 * n] = 1.0
    lower[2 * n :] = -highspy.kHighsInf
    upper[2 * n :] = 1.0

    h.addRows(
        num_rows,
        lower,
        upper,
        total_nz,
        starts[:-1],  # HiGHS takes starts without sentinel
        all_idx,
        all_val,
    )

    t_build = time.perf_counter() - t0

    h.setOptionValue("time_limit", max_solve_sec)
    t1 = time.perf_counter()
    h.run()
    t_solve = time.perf_counter() - t1

    ms = h.getModelStatus()
    return t_build, t_solve, h.modelStatusToString(ms)


# ── main ─────────────────────────────────────────────────────────────────────

def detect_solvers():
    solvers = []
    try:
        import mip
        try:
            m = mip.Model(solver_name=mip.CBC)
            m.verbose = 0
            m.add_var()
            m.optimize()
            solvers.append("CBC")
        except Exception:
            pass
        try:
            m = mip.Model(solver_name=mip.HIGHS)
            m.verbose = 0
            m.add_var()
            m.optimize()
            solvers.append("HIGHS")
        except Exception:
            pass
        try:
            m = mip.Model(solver_name=mip.GUROBI)
            m.verbose = 0
            m.add_var()
            m.optimize()
            solvers.append("GUROBI")
        except Exception:
            pass
    except ImportError:
        pass
    try:
        import highspy
        solvers.append("highspy-hl")
        solvers.append("highspy-batch")
    except ImportError:
        pass
    return solvers


if __name__ == "__main__":
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    build_only = "--build-only" in flags
    size_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sizes = [int(a) for a in size_args] if size_args else SIZES
    max_solve_sec = 5.0

    py_impl = platform.python_implementation()
    py_ver = platform.python_version()
    print(f"\nN-Queens benchmark — {py_impl} {py_ver}")
    print(f"Sizes: {sizes}   (solve limit: {max_solve_sec}s)")
    if build_only:
        print("Mode: build-only (no solve)")

    solvers = detect_solvers()
    print(f"Solvers detected: {', '.join(solvers)}\n")

    for n in sizes:
        hline()
        print(f"  n = {n}  ({n}x{n} = {n*n} binary variables)")
        hline()
        print(f"  {'Solver':<28}", end="")
        if build_only:
            print(f"  {'build (s)':>10}")
        else:
            print(f"  {'build (s)':>10}  {'solve (s)':>10}  {'status':>16}")
        hline()

        for solver in solvers:
            try:
                if solver in ("CBC", "HIGHS", "GUROBI"):
                    tb, ts, st = bench_pmip(n, solver, max_solve_sec)
                elif solver == "highspy-hl":
                    tb, ts, st = bench_highspy_hl(n, max_solve_sec)
                elif solver == "highspy-batch":
                    tb, ts, st = bench_highspy_batch(n, max_solve_sec)
                else:
                    continue
                if build_only:
                    print(f"  {solver:<28}  {tb:>10.3f}")
                else:
                    print(f"  {solver:<28}  {tb:>10.3f}  {ts:>10.3f}  {st:>16}")
            except Exception as e:
                print(f"  {solver:<28}  ERROR: {e}")

    hline()
    print()
