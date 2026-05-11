"""
RCPSP (Resource-Constrained Project Scheduling Problem) model-build benchmark.

Generates random RCPSP instances with varying numbers of jobs, resources, and
precedence densities, then measures how long each python-mip backend takes to
build (but not solve) the MIP model.

Usage:
    python benchmarks/rcpsp_bench.py [--build-only] [--verify]

--build-only   Measure model creation time only (default behaviour).
--verify       Build and solve the smallest instance with CBC to check
               that the formulation is correct, then exit.
"""

import argparse
import random
import signal
import sys
import time
from itertools import product

# ── timeout helper (Linux only) ────────────────────────────────────────────────

BUILD_TIMEOUT_SEC = 8


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def _run_with_timeout(fn, *args, **kwargs):
    """Return (elapsed, result) or ('>8s', None) on timeout."""
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(BUILD_TIMEOUT_SEC)
    try:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        signal.alarm(0)
        return elapsed, result
    except _Timeout:
        return f">{BUILD_TIMEOUT_SEC}s", None
    finally:
        signal.alarm(0)


# ── random instance generator ──────────────────────────────────────────────────

def make_rcpsp(n_jobs, n_resources, n_prec, p_range=(1, 5), c_range=(4, 8), seed=42):
    """
    Generate a random RCPSP instance.

    Returns (p, u, c, S) where:
      p[j]    – processing time of job j  (index 0 and n+1 are dummy jobs)
      u[j][r] – resource r consumed by job j while executing
      c[r]    – capacity of resource r
      S       – list of [pred, succ] precedence pairs (0-indexed, inclusive of dummies)
    """
    rng = random.Random(seed)
    n_total = n_jobs + 2  # real jobs 1..n_jobs; dummy 0 and n_jobs+1

    # Processing times
    p = [0] + [rng.randint(*p_range) for _ in range(n_jobs)] + [0]

    # Resource usage (0 when no resource needed; dummies use nothing)
    u = [[0] * n_resources]
    for _ in range(n_jobs):
        row = [rng.randint(0, max(1, c_range[0] // 2)) for _ in range(n_resources)]
        u.append(row)
    u.append([0] * n_resources)

    # Resource capacities
    c = [rng.randint(*c_range) for _ in range(n_resources)]

    # Precedences: dummy 0 → all real jobs, all real jobs → dummy n+1
    S = [[0, j] for j in range(1, n_jobs + 1)]
    S += [[j, n_jobs + 1] for j in range(1, n_jobs + 1)]

    # Extra random precedences among real jobs (forward arcs only to avoid cycles)
    added = set()
    attempts = 0
    while len(added) < n_prec and attempts < n_prec * 20:
        i = rng.randint(1, n_jobs - 1)
        j = rng.randint(i + 1, n_jobs)
        if (i, j) not in added:
            added.add((i, j))
            S.append([i, j])
        attempts += 1

    return p, u, c, S


# ── model builders ─────────────────────────────────────────────────────────────

def build_model(solver_name, p, u, c, S, build_only=True):
    """Build the RCPSP MIP model and return the Model object."""
    from mip import Model, xsum, BINARY

    R = range(len(c))
    J = range(len(p))
    T = range(sum(p))
    n = len(p) - 2  # number of real jobs

    m = Model(solver_name=solver_name)
    m.verbose = 0

    x = [[m.add_var(name=f"x({j},{t})", var_type=BINARY) for t in T] for j in J]

    m.objective = xsum(t * x[n + 1][t] for t in T)

    for j in J:
        m += xsum(x[j][t] for t in T) == 1

    for r, t in product(R, T):
        m += (
            xsum(
                u[j][r] * x[j][t2]
                for j in J
                for t2 in range(max(0, t - p[j] + 1), t + 1)
            )
            <= c[r]
        )

    for pred, succ in S:
        m += xsum(t * x[succ][t] - t * x[pred][t] for t in T) >= p[pred]

    return m


def bench_pmip(solver_name, p, u, c, S, build_only=True):
    build_model(solver_name, p, u, c, S, build_only)


# ── native gurobipy benchmark ─────────────────────────────────────────────────

def bench_gurobi_native(p, u, c, S, build_only=True):
    """Build RCPSP with the native gurobipy API (no python-mip layer)."""
    import gurobipy as gp
    from gurobipy import GRB

    R = range(len(c))
    J = range(len(p))
    T = range(sum(p))
    n = len(p) - 2

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(env=env)

    x = m.addVars([(j, t) for j in J for t in T], vtype=GRB.BINARY)

    m.setObjective(gp.quicksum(t * x[n + 1, t] for t in T), GRB.MINIMIZE)

    for j in J:
        m.addConstr(gp.quicksum(x[j, t] for t in T) == 1)

    for r, t in product(R, T):
        m.addConstr(
            gp.quicksum(
                u[j][r] * x[j, t2]
                for j in J
                for t2 in range(max(0, t - p[j] + 1), t + 1)
            )
            <= c[r]
        )

    for pred, succ in S:
        m.addConstr(
            gp.quicksum(t * (x[succ, t] - x[pred, t]) for t in T) >= p[pred]
        )

    m.update()


# ── benchmark configurations ───────────────────────────────────────────────────

CONFIGS = [
    dict(name="2R / sparse prec.", n_resources=2, prec_factor=1, p_range=(1, 4)),
    dict(name="2R / dense prec.",  n_resources=2, prec_factor=3, p_range=(1, 4)),
    dict(name="4R / sparse prec.", n_resources=4, prec_factor=1, p_range=(1, 4)),
    dict(name="4R / dense prec.",  n_resources=4, prec_factor=3, p_range=(1, 4)),
]

SIZES = [10, 20, 30, 50, 75, 100, 150, 200]


# ── main ────────────────────────────────────────────────────────────────────────

def run_benchmarks(build_only=True):
    from mip.constants import CBC, HIGHS

    import mip.highs as _h
    has_highs = _h.has_highs

    try:
        from mip.constants import GUROBI
        import gurobipy  # noqa: F401
        has_gurobi = True
    except Exception:
        has_gurobi = False
        GUROBI = None

    col_w = 16

    for cfg in CONFIGS:
        n_res = cfg["n_resources"]
        prec_factor = cfg["prec_factor"]
        p_range = cfg["p_range"]

        print(f"\n=== Config: {cfg['name']} ===")
        print(f"{'Jobs':<6}", end="")
        print(f"{'CBC':>{col_w}}", end="")
        if has_highs:
            print(f"{'HiGHS':>{col_w}}", end="")
        if has_gurobi:
            print(f"{'python-mip/Gurobi':>{col_w}}", end="")
            print(f"{'gurobipy':>{col_w}}", end="")
        print()

        for n_jobs in SIZES:
            n_prec = prec_factor * n_jobs
            p, u, c, S = make_rcpsp(n_jobs, n_res, n_prec, p_range=p_range)
            T = sum(p)
            n_vars = (n_jobs + 2) * T

            row = f"{n_jobs:<6}"

            def fmt(elapsed):
                return str(round(elapsed, 3) if isinstance(elapsed, float) else elapsed)

            # CBC
            elapsed, _ = _run_with_timeout(bench_pmip, CBC, p, u, c, S, build_only)
            row += f"{fmt(elapsed):>{col_w}}"

            # HiGHS via python-mip
            if has_highs:
                elapsed, _ = _run_with_timeout(bench_pmip, HIGHS, p, u, c, S, build_only)
                row += f"{fmt(elapsed):>{col_w}}"

            # Gurobi via python-mip
            if has_gurobi:
                elapsed, _ = _run_with_timeout(bench_pmip, GUROBI, p, u, c, S, build_only)
                row += f"{fmt(elapsed):>{col_w}}"

                # Gurobi native gurobipy
                elapsed, _ = _run_with_timeout(bench_gurobi_native, p, u, c, S, build_only)
                row += f"{fmt(elapsed):>{col_w}}"

            row += f"   (n_vars={n_vars}, T={T})"
            print(row)


def verify():
    """Build and solve the smallest instance to check correctness."""
    p, u, c, S = make_rcpsp(10, 2, 10, p_range=(1, 4))
    from mip.constants import CBC
    m = build_model(CBC, p, u, c, S, build_only=False)
    m.verbose = 1
    m.optimize()
    print(f"\nStatus: {m.status}  Objective: {m.objective_value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RCPSP model-build benchmark")
    parser.add_argument("--build-only", action="store_true", default=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        verify()
    else:
        run_benchmarks(build_only=True)
