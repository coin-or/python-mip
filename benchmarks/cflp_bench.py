"""Capacitated Facility Location Problem (CFLP) model-building benchmark.

A classic applied supply-chain / logistics problem: decide which facilities
to open and how to serve customer demand from them at minimum cost, subject
to facility capacity.

  Variables:
    y[i]   ∈ {0,1}      — facility i is opened
    x[i,j] ∈ [0,1]       — fraction of customer j's demand served by facility i

  min   Σ_i f[i]·y[i] + Σ_{i,j} c[i,j]·d[j]·x[i,j]
  s.t.  Σ_i x[i,j] = 1                      ∀j        (demand fully served)
        Σ_j d[j]·x[i,j] ≤ cap[i]·y[i]       ∀i        (aggregated capacity)

Like n-Queens, this formulation has O(n²) variables (the assignment matrix
x) but only O(n) constraints (2n: one demand row per customer, one
aggregated capacity row per facility) — the same "many variables, few
constraints" profile, but for a widely-used applied combinatorial
optimisation problem (facility/warehouse location, e.g. the classic
ORLIB-style capacitated location instances) rather than a puzzle.

Random Euclidean instances (facilities and customers scattered in a square)
are generated with a fixed seed for reproducibility across interpreters and
solver backends. The number of facilities equals the number of customers
(both set to n) so that instance size is controlled by a single parameter,
matching the n-Queens benchmark's size range.

Usage:
    python cflp_bench.py [sizes]          # e.g. 200 400 600 800 1000 1200
    python cflp_bench.py --build-only     # skip solve timing
    python cflp_bench.py --verify         # solve small instance only
"""

import math
import platform
import random
import signal
import sys
import time

SIZES = [200, 400, 600, 800, 1000, 1200]
SEED = 42
VERIFY_N = 15
MAX_SOLVE_SEC = 30.0
BUILD_TIMEOUT_SEC = 8  # seconds; slower solvers print ">8s" and skip

CAPACITY_SLACK = 1.5  # total capacity ≈ CAPACITY_SLACK × total demand


# ── build timeout helper ─────────────────────────────────────────────────────


class _BuildTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _BuildTimeout()


def _run_with_timeout(fn, timeout_sec=BUILD_TIMEOUT_SEC):
    """Call fn() and return its result; raise _BuildTimeout if too slow."""
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ── instance generation ──────────────────────────────────────────────────────


def make_instance(n, seed=SEED):
    """Random Euclidean CFLP: n facilities and n customers in [0, 1000]².

    Returns (c, f, d, cap) where:
      c[i,j] — unit transportation cost (Euclidean distance) facility i→customer j
      f[i]   — fixed cost of opening facility i
      d[j]   — demand of customer j
      cap[i] — capacity of facility i
    """
    rng = random.Random(seed)
    fac_pts = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]
    cus_pts = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]

    c = {}
    for i in range(n):
        xi, yi = fac_pts[i]
        for j in range(n):
            dx = xi - cus_pts[j][0]
            dy = yi - cus_pts[j][1]
            c[i, j] = math.sqrt(dx * dx + dy * dy) / 100.0

    f = [rng.uniform(500, 1500) for _ in range(n)]
    d = [rng.uniform(1, 10) for _ in range(n)]
    total_d = sum(d)
    avg_cap = (total_d * CAPACITY_SLACK) / n
    cap = [avg_cap * rng.uniform(0.8, 1.2) for _ in range(n)]
    return c, f, d, cap


# ── CSR builder for highspy batch API ────────────────────────────────────────


def _cflp_csr(n, d, cap):
    """Build CSR row data for demand + aggregated capacity constraints.

    Variable layout in HiGHS:
      cols 0 .. n*n-1     : x[i,j] continuous vars, row-major (i*n + j)
      cols n*n .. n*n+n-1 : y[i] binary vars

    Returns (starts, indices, values, lb, ub) as numpy arrays.
    Row order: demand (n), capacity (n).
    """
    import numpy as np

    nn = n * n
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

    # demand: Σ_i x[i,j] = 1
    for j in range(n):
        idx = [i * n + j for i in range(n)]
        add_row(idx, [1.0] * n, 1.0, 1.0)

    # aggregated capacity: Σ_j d[j]·x[i,j] − cap[i]·y[i] ≤ 0
    for i in range(n):
        idx = [i * n + j for j in range(n)] + [nn + i]
        val = [d[j] for j in range(n)] + [-cap[i]]
        add_row(idx, val, -1e30, 0.0)

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

    c, f, d, cap = make_instance(n)
    I = range(n)
    J = range(n)

    t0 = time.perf_counter()
    m = Model(solver_name=solver_name)
    m.verbose = 0

    y = [m.add_var(var_type=BINARY) for _ in I]
    x = {(i, j): m.add_var(lb=0.0, ub=1.0) for i in I for j in J}

    m.objective = minimize(
        xsum(f[i] * y[i] for i in I)
        + xsum(c[i, j] * d[j] * x[i, j] for i in I for j in J)
    )

    for j in J:
        m += xsum(x[i, j] for i in I) == 1
    for i in I:
        m += xsum(d[j] * x[i, j] for j in J) <= cap[i] * y[i]

    t_build = time.perf_counter() - t0

    if build_only:
        return t_build, None, "—"

    m.max_seconds = max_solve_sec
    t1 = time.perf_counter()
    status = m.optimize()
    t_solve = time.perf_counter() - t1
    obj = m.objective_value if m.num_solutions else None
    return t_build, t_solve, str(status).split(".")[-1], obj


# ── native gurobipy benchmark ─────────────────────────────────────────────────


def bench_gurobi_native(n, build_only=False, max_solve_sec=MAX_SOLVE_SEC):
    """Build CFLP model with the native gurobipy API (no python-mip layer)."""
    import gurobipy as gp
    from gurobipy import GRB

    c, f, d, cap = make_instance(n)
    I = range(n)
    J = range(n)

    t0 = time.perf_counter()
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(env=env)

    y = [m.addVar(vtype=GRB.BINARY) for _ in I]
    x = {(i, j): m.addVar(lb=0.0, ub=1.0) for i in I for j in J}

    m.setObjective(
        gp.quicksum(f[i] * y[i] for i in I)
        + gp.quicksum(c[i, j] * d[j] * x[i, j] for i in I for j in J),
        GRB.MINIMIZE,
    )

    for j in J:
        m.addConstr(gp.quicksum(x[i, j] for i in I) == 1)
    for i in I:
        m.addConstr(gp.quicksum(d[j] * x[i, j] for j in J) <= cap[i] * y[i])

    m.update()
    t_build = time.perf_counter() - t0

    if build_only:
        return t_build, None, "—", None

    m.setParam("TimeLimit", max_solve_sec)
    t1 = time.perf_counter()
    m.optimize()
    t_solve = time.perf_counter() - t1
    obj = m.ObjVal if m.SolCount > 0 else None
    return t_build, t_solve, str(m.Status), obj


# ── highspy high-level benchmark ─────────────────────────────────────────────


def bench_highspy_hl(n, build_only=False, max_solve_sec=MAX_SOLVE_SEC):
    """highspy using its recommended vectorized numpy-array API
    (addVariables/addBinaries/addConstrs on dense arrays)."""
    import highspy
    import numpy as np

    c, f, d, cap = make_instance(n)
    d_arr = np.array(d, dtype=np.float64)
    cap_arr = np.array(cap, dtype=np.float64)
    f_arr = np.array(f, dtype=np.float64)
    c_dense = np.array([[c[i, j] for j in range(n)] for i in range(n)],
                        dtype=np.float64)

    t0 = time.perf_counter()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)

    y = h.addBinaries(n)
    h.changeColsCost(n, np.arange(n, dtype=np.int32), f_arr)

    x = h.addVariables(n, n, lb=0, ub=1)
    x_cost = c_dense * d_arr[np.newaxis, :]
    h.changeColsCost(n * n, np.arange(n, n + n * n, dtype=np.int32), x_cost.flatten())

    h.addConstrs(x.sum(axis=0) == 1)
    h.addConstrs((x * d_arr[np.newaxis, :]).sum(axis=1) <= cap_arr * y)

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

    c, f, d, cap = make_instance(n)
    nn = n * n

    t0 = time.perf_counter()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)

    # x[i,j] continuous vars, row-major (i*n + j), cost = c[i,j]*d[j]
    x_costs = np.array([c[i, j] * d[j] for i in range(n) for j in range(n)],
                        dtype=np.float64)
    x_lb = np.zeros(nn, dtype=np.float64)
    x_ub = np.ones(nn, dtype=np.float64)
    h.addCols(
        nn, x_costs, x_lb, x_ub,
        0, np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64),
    )

    # y[i] binary vars, cost = f[i]
    y_costs = np.array(f, dtype=np.float64)
    y_lb = np.zeros(n, dtype=np.float64)
    y_ub = np.ones(n, dtype=np.float64)
    h.addCols(
        n, y_costs, y_lb, y_ub,
        0, np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64),
    )
    kInteger = highspy.HighsVarType.kInteger
    h.changeColsIntegrality(
        n,
        np.arange(nn, nn + n, dtype=np.int32),
        np.full(n, kInteger, dtype=np.uint8),
    )

    starts, idx, val, lb, ub = _cflp_csr(n, d, cap)
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
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        gp.Model(env=env)
        solvers.append("gurobipy")
    except Exception:
        pass
    try:
        import highspy  # noqa: F401

        solvers.append("highspy-hl")
        solvers.append("highspy-batch")
    except ImportError:
        pass
    return solvers


def _model_size(n):
    nvars = n * n + n
    nrows = 2 * n
    return nvars, nrows


def _run(solver, n, build_only, max_solve_sec):
    if solver in ("CBC", "HIGHS", "GUROBI"):
        return bench_pmip(n, solver, build_only, max_solve_sec)
    elif solver == "gurobipy":
        return bench_gurobi_native(n, build_only, max_solve_sec)
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
    print(f"\nCapacitated Facility Location benchmark — {py_impl} {py_ver}")
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
        hline()
        print(
            f"  n = {n}  |  facilities = {n}  |  customers = {n}  "
            f"|  vars = {nvars}  |  constraints = {nrows}"
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
                def _bench(s=solver, _n=n):
                    return _run(s, _n, build_only, MAX_SOLVE_SEC)
                try:
                    result = _run_with_timeout(_bench)
                except _BuildTimeout:
                    print(f"  {solver:<32}  {f'>{BUILD_TIMEOUT_SEC}s':>10}")
                    continue
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
                        assert abs(obj - prev_obj) < max(1.0, 0.01 * abs(prev_obj)), (
                            f"Obj mismatch: {solver}={obj:.2f} vs {prev_obj:.2f}"
                        )
                    if obj is not None:
                        prev_obj = obj
            except Exception as e:
                print(f"  {solver:<32}  ERROR: {e}")

    hline()
    print()
