"""TSP single-commodity flow (Gavish-Graves) model-building benchmark.

Compares model creation times across solvers and Python interpreters using
the compact flow formulation for the Traveling Salesman Problem:

  Variables:
    x[i,j] ∈ {0,1}   — arc (i,j) is used in the tour
    y[i,j] ≥ 0       — flow units routed on arc (i,j)

  min   Σ_{i≠j} c[i,j] · x[i,j]
  s.t.  Σ_j x[i,j] = 1               ∀i        (leave each city once)
        Σ_i x[i,j] = 1               ∀j        (enter each city once)
        y[i,j] ≤ (n-1) · x[i,j]     ∀i≠j      (flow only on used arcs)
        Σ_j y[j,i] − Σ_j y[i,j] = 1 ∀i=1..n-1 (flow conservation)

  City 0 is the depot; each non-depot city consumes one unit of flow.
  The capacity + flow constraints eliminate subtours without cutting planes.

Random Euclidean instances are generated with a fixed seed for
reproducibility across interpreters and solver backends.

Usage:
    python tsp_flow_bench.py [sizes]          # e.g. 15 20 30 50
    python tsp_flow_bench.py --build-only     # skip solve timing
    python tsp_flow_bench.py --verify         # solve small instance only
"""

import math
import platform
import random
import sys
import time

SIZES = [15, 20, 30, 50]
SEED = 42
VERIFY_N = 10        # solve to optimality and check
MAX_SOLVE_SEC = 30.0


# ── instance generation ──────────────────────────────────────────────────────


def make_instance(n, seed=SEED):
    """Random Euclidean TSP on n cities in [0, 1000]²."""
    rng = random.Random(seed)
    pts = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]
    c = {}
    for i in range(n):
        xi, yi = pts[i]
        for j in range(n):
            if i != j:
                dx = xi - pts[j][0]
                dy = yi - pts[j][1]
                c[i, j] = math.sqrt(dx * dx + dy * dy)
    return c


def _arc_list(n):
    return [(i, j) for i in range(n) for j in range(n) if i != j]


# ── CSR builder for highspy batch API ────────────────────────────────────────


def _tsp_flow_csr(n):
    """Build CSR row data for all TSP flow constraints.

    Variable layout in HiGHS:
      cols 0 .. na-1      : x[k] binary arc vars  (cost c[i,j])
      cols na .. 2*na-1   : y[k] continuous flow vars  [0, n-1]

    Returns (starts, indices, values, lb, ub) as numpy arrays.
    Row order: out-degree (n), in-degree (n), capacity (na), flow (n-1).
    """
    import numpy as np

    A = _arc_list(n)
    na = len(A)
    arc_idx = {a: k for k, a in enumerate(A)}
    INF = 1e30

    row_idx = []
    row_val = []
    row_lb = []
    row_ub = []
    starts = []
    nz = 0

    def add_row(idx, val, lb, ub):
        nonlocal nz
        starts.append(nz)
        row_idx.extend(idx)
        row_val.extend(val)
        row_lb.append(lb)
        row_ub.append(ub)
        nz += len(idx)

    # out-degree: Σ_j x[i,j] = 1
    for i in range(n):
        idx = [arc_idx[i, j] for j in range(n) if j != i]
        add_row(idx, [1.0] * (n - 1), 1.0, 1.0)

    # in-degree: Σ_i x[i,j] = 1
    for j in range(n):
        idx = [arc_idx[i, j] for i in range(n) if i != j]
        add_row(idx, [1.0] * (n - 1), 1.0, 1.0)

    # capacity: y[k] - (n-1)*x[k] ≤ 0
    for k in range(na):
        add_row([na + k, k], [1.0, -(n - 1)], -INF, 0.0)

    # flow conservation: Σ_j y[j,i] - Σ_j y[i,j] = 1  for i=1..n-1
    for i in range(1, n):
        idx = []
        val = []
        for j in range(n):
            if j != i:
                idx.append(na + arc_idx[j, i])   # inflow: +1
                val.append(1.0)
                idx.append(na + arc_idx[i, j])   # outflow: -1
                val.append(-1.0)
        add_row(idx, val, 1.0, 1.0)

    return (
        np.array(starts, dtype=np.int32),
        np.array(row_idx, dtype=np.int32),
        np.array(row_val, dtype=np.float64),
        np.array(row_lb, dtype=np.float64),
        np.array(row_ub, dtype=np.float64),
    )


# ── python-mip benchmark ─────────────────────────────────────────────────────


def bench_pmip(n, solver_name, build_only=False, max_solve_sec=MAX_SOLVE_SEC):
    import mip
    from mip import BINARY, Model, minimize, xsum

    c = make_instance(n)
    V = range(n)
    A = _arc_list(n)

    t0 = time.perf_counter()
    m = Model(solver_name=solver_name)
    m.verbose = 0

    x = {ij: m.add_var(var_type=BINARY) for ij in A}
    y = {ij: m.add_var(lb=0.0, ub=n - 1) for ij in A}

    m.objective = minimize(xsum(c[i, j] * x[i, j] for i, j in A))

    for i in V:
        m += xsum(x[i, j] for j in V if j != i) == 1
        m += xsum(x[j, i] for j in V if j != i) == 1
    for i, j in A:
        m += y[i, j] <= (n - 1) * x[i, j]
    for i in range(1, n):
        m += (
            xsum(y[j, i] for j in V if j != i)
            - xsum(y[i, j] for j in V if j != i)
            == 1
        )

    t_build = time.perf_counter() - t0

    if build_only:
        return t_build, None, "—"

    m.max_seconds = max_solve_sec
    t1 = time.perf_counter()
    status = m.optimize()
    t_solve = time.perf_counter() - t1
    obj = m.objective_value if m.num_solutions else None
    return t_build, t_solve, str(status).split(".")[-1], obj


# ── highspy high-level benchmark ─────────────────────────────────────────────


def bench_highspy_hl(n, build_only=False, max_solve_sec=MAX_SOLVE_SEC):
    """highspy using high-level addVariable/addConstr expression API."""
    import highspy

    c = make_instance(n)
    V = range(n)
    A = _arc_list(n)
    na = len(A)
    arc_idx = {a: k for k, a in enumerate(A)}
    kInteger = highspy.HighsVarType.kInteger

    t0 = time.perf_counter()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)

    # binary arc vars with costs
    xv = [h.addVariable(lb=0, ub=1) for _ in A]
    for k, (i, j) in enumerate(A):
        h.changeColIntegrality(k, kInteger)
        h.changeColCost(k, c[i, j])

    # continuous flow vars [0, n-1]
    yv = [h.addVariable(lb=0, ub=n - 1) for _ in A]

    for i in V:
        h.addConstr(sum(xv[arc_idx[i, j]] for j in V if j != i) == 1)
        h.addConstr(sum(xv[arc_idx[j, i]] for j in V if j != i) == 1)
    for k in range(na):
        h.addConstr(yv[k] - (n - 1) * xv[k] <= 0)
    for i in range(1, n):
        h.addConstr(
            sum(yv[arc_idx[j, i]] for j in V if j != i)
            - sum(yv[arc_idx[i, j]] for j in V if j != i)
            == 1
        )

    t_build = time.perf_counter() - t0

    if build_only:
        return t_build, None, "—", None

    h.setOptionValue("time_limit", max_solve_sec)
    t1 = time.perf_counter()
    h.run()
    t_solve = time.perf_counter() - t1
    ms = h.getModelStatus()
    _, obj = h.getInfoValue("objective_function_value")
    return t_build, t_solve, h.modelStatusToString(ms), obj


# ── highspy batch (numpy) benchmark ──────────────────────────────────────────


def bench_highspy_batch(n, build_only=False, max_solve_sec=MAX_SOLVE_SEC):
    """highspy using batch numpy addCols/addRows API (CSR layout)."""
    import highspy
    import numpy as np

    c = make_instance(n)
    A = _arc_list(n)
    na = len(A)

    t0 = time.perf_counter()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)

    # x arc binary vars: addCols passes costs directly
    x_costs = np.array([c[a] for a in A], dtype=np.float64)
    x_lb = np.zeros(na, dtype=np.float64)
    x_ub = np.ones(na, dtype=np.float64)
    h.addCols(
        na, x_costs, x_lb, x_ub,
        0, np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64),
    )
    kInteger = highspy.HighsVarType.kInteger
    h.changeColsIntegrality(
        na,
        np.arange(na, dtype=np.int32),
        np.full(na, kInteger, dtype=np.uint8),
    )

    # y flow vars: continuous [0, n-1]
    y_lb = np.zeros(na, dtype=np.float64)
    y_ub = np.full(na, float(n - 1), dtype=np.float64)
    h.addVars(na, y_lb, y_ub)

    # all constraints in one CSR batch call
    starts, idx, val, lb, ub = _tsp_flow_csr(n)
    nrows = len(starts)
    h.addRows(nrows, lb, ub, len(idx), starts, idx, val)

    t_build = time.perf_counter() - t0

    if build_only:
        return t_build, None, "—", None

    h.setOptionValue("time_limit", max_solve_sec)
    t1 = time.perf_counter()
    h.run()
    t_solve = time.perf_counter() - t1
    ms = h.getModelStatus()
    _, obj = h.getInfoValue("objective_function_value")
    return t_build, t_solve, h.modelStatusToString(ms), obj


# ── helpers ──────────────────────────────────────────────────────────────────


def hline():
    print("-" * 80)


def row(label, *vals):
    print(f"  {label:<32}", end="")
    for v in vals:
        print(f"  {v:>10}", end="")
    print()


def detect_solvers():
    solvers = []
    try:
        import mip

        for name in ("CBC", "HIGHS", "GUROBI"):
            try:
                m = mip.Model(solver_name=name)
                m.verbose = 0
                m.add_var()
                m.optimize()
                solvers.append(name)
            except Exception:
                pass
    except ImportError:
        pass
    try:
        import highspy  # noqa: F401

        solvers.append("highspy-hl")
        solvers.append("highspy-batch")
    except ImportError:
        pass
    return solvers


def _model_size(n):
    na = n * (n - 1)
    nvars = 2 * na
    nrows = 2 * n + na + (n - 1)
    return nvars, nrows


def _run(solver, n, build_only, max_solve_sec):
    if solver in ("CBC", "HIGHS", "GUROBI"):
        return bench_pmip(n, solver, build_only, max_solve_sec)
    elif solver == "highspy-hl":
        return bench_highspy_hl(n, build_only, max_solve_sec)
    elif solver == "highspy-batch":
        return bench_highspy_batch(n, build_only, max_solve_sec)
    raise ValueError(solver)


# ── main ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    args = sys.argv[1:]
    build_only = "--build-only" in args
    verify = "--verify" in args
    size_args = [a for a in args if not a.startswith("--")]
    sizes = [int(a) for a in size_args] if size_args else SIZES

    py_impl = platform.python_implementation()
    py_ver = platform.python_version()
    print(f"\nTSP single-commodity flow benchmark — {py_impl} {py_ver}")
    print(f"Seed: {SEED}  |  Solve limit: {MAX_SOLVE_SEC}s")
    if build_only:
        print("Mode: build-only (no solve)")
    print()

    solvers = detect_solvers()
    print(f"Solvers detected: {', '.join(solvers)}\n")

    if verify:
        sizes = [VERIFY_N]

    for n in sizes:
        nvars, nrows = _model_size(n)
        na = n * (n - 1)
        hline()
        print(
            f"  n = {n}  |  arcs = {na}  |  vars = {nvars}  "
            f"|  constraints = {nrows}"
        )
        hline()
        print(f"  {'Solver':<32}", end="")
        if build_only:
            print(f"  {'build (s)':>10}")
        else:
            print(f"  {'build (s)':>10}  {'solve (s)':>10}  {'status':>16}  {'obj':>10}")

        hline()
        prev_obj = None

        for solver in solvers:
            try:
                result = _run(solver, n, build_only, MAX_SOLVE_SEC)
                tb = result[0]
                if build_only:
                    print(f"  {solver:<32}  {tb:>10.3f}")
                else:
                    _, ts, st, obj = result
                    obj_s = f"{obj:.2f}" if obj is not None else "—"
                    print(
                        f"  {solver:<32}  {tb:>10.3f}  {ts:>10.3f}"
                        f"  {st:>16}  {obj_s:>10}"
                    )
                    # cross-check objective between solvers
                    if obj is not None and prev_obj is not None:
                        assert abs(obj - prev_obj) < 0.5, (
                            f"Obj mismatch: {solver}={obj:.2f} vs {prev_obj:.2f}"
                        )
                    if obj is not None:
                        prev_obj = obj
            except Exception as e:
                print(f"  {solver:<32}  ERROR: {e}")

    hline()
    print()
