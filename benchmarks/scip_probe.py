"""Native PySCIPOpt feasibility and model-building probes (not a mip backend).

Install with ``python -m pip install -e ".[scip,highs,test]"`` and run:

    python -m benchmarks.scip_probe --checks-only
    python -m benchmarks.scip_probe --sizes 1000 10000 --widths 8 128 --repeats 3
    python -m benchmarks.scip_probe --sizes 1000 --solve
    python -m benchmarks.scip_probe --sizes 1000 --widths 128 --solve --scip-no-presolve

Build timings include expression construction and insertion, including any
buffer flush, but exclude deterministic input generation. SCIP comparisons use
the same native API except for the row-building strategy. Python-MIP comparisons
include its modeling layer and any native buffer flush; they are not pure
native-API comparisons. All solvers use one thread and bounded solve time.
Ring constraints use the nonunit coefficient 1.5, exercising coefficient
multiplication rather than just sums of bare variables.

``terms`` builds Expr dictionaries using pyscipopt.scip.Term, avoiding temporary
expressions for each coefficient. ``addconss`` buffers these same expressions
before calling addConss; that API still inserts constraints individually.
``coefficients`` inserts an empty linear constraint and adds each coefficient.
The small LP dual probes disable presolve, heuristics, and propagation to retain
original rows. An unchanged re-solve can still make duals unavailable; a backend
must check/cache them rather than expose SCIP's invalid-value sentinel as a dual.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import platform
from statistics import median
from time import perf_counter

import pyscipopt
from pyscipopt import Expr, Model, SCIP_PARAMSETTING, quicksum
from pyscipopt.scip import Term

STRATEGIES = ("quicksum", "terms", "addconss", "coefficients")
ROW_COEFFICIENT = 1.5


@dataclass(frozen=True)
class Solution:
    status: str
    objective: float | None
    values: tuple[float, ...]
    bound: float
    num_solutions: int


@dataclass(frozen=True)
class BuildTiming:
    setup: float
    variables: float
    constraints: float
    flush: float
    solve: float
    status: str

    @property
    def total(self) -> float:
        return self.setup + self.variables + self.constraints + self.flush


def new_model(name: str, *, lp_duals: bool = False) -> Model:
    model = Model(name)
    model.hideOutput()
    model.setIntParam("parallel/maxnthreads", 1)
    model.setRealParam("limits/time", 10.0)
    if lp_duals:
        model.setPresolve(SCIP_PARAMSETTING.OFF)
        model.setHeuristics(SCIP_PARAMSETTING.OFF)
        model.disablePropagation()
    return model


def solution(model: Model, variables) -> Solution:
    """Copy primal values before freeTransform invalidates solution handles."""
    status = str(model.getStatus())
    count = model.getNSols()
    usable = count > 0 and status not in ("infeasible", "unbounded", "inforunbd")
    best = model.getBestSol() if usable else None
    return Solution(
        status,
        model.getSolObjVal(best) if usable else None,
        tuple(model.getSolVal(best, var) for var in variables) if usable else (),
        model.getDualbound(),
        count,
    )


def build_lp(*, maximize: bool = True):
    model = new_model("lp", lp_duals=True)
    x = model.addVar("x", obj=3.0)
    y = model.addVar("y", obj=2.0)
    capacity = model.addCons(2 * x + y <= 4, name="capacity")
    balance = model.addCons(x + 2 * y <= 4, name="balance")
    model.setObjective(
        3 * x + 2 * y if maximize else -3 * x - 2 * y,
        sense="maximize" if maximize else "minimize",
    )
    return model, (x, y), (capacity, balance)


def build_milp():
    """The mixed C/I/B model from test_model.py::test_solve_relaxation."""
    model = new_model("mixed")
    x = model.addVar("x")
    y = model.addVar("y", vtype="I")
    z = model.addVar("z", vtype="B")
    model.addCons(x <= 10 * z)
    model.addCons(x <= 9.5)
    model.addCons(x + y <= 20)
    model.setObjective(4 * x + y - z, sense="maximize")
    return model, (x, y, z)


def build_knapsack():
    """The model and data from examples/knapsack.py."""
    profits = (10, 13, 18, 31, 7, 15)
    weights = (11, 15, 20, 35, 10, 33)
    model = new_model("knapsack")
    variables = tuple(
        model.addVar(f"x{i}", vtype="B", obj=profit) for i, profit in enumerate(profits)
    )
    capacity = model.addCons(quicksum(w * x for w, x in zip(weights, variables)) <= 47)
    model.setMaximize()
    return model, variables, capacity


def build_queens(n: int = 8):
    model = new_model("queens")
    variables = tuple(model.addVar(f"x{i}", vtype="B") for i in range(n * n))
    for i in range(n):
        model.addCons(quicksum(variables[i * n + j] for j in range(n)) == 1)
        model.addCons(quicksum(variables[j * n + i] for j in range(n)) == 1)
    for k in range(2 - n, n - 1):
        cells = [variables[i * n + i - k] for i in range(n) if 0 <= i - k < n]
        if len(cells) >= 2:
            model.addCons(quicksum(cells) <= 1)
    for k in range(1, 2 * n - 2):
        cells = [variables[i * n + k - i] for i in range(n) if 0 <= k - i < n]
        if len(cells) >= 2:
            model.addCons(quicksum(cells) <= 1)
    return model, variables


def check_small_models() -> None:
    model, variables, constraints = build_lp()
    model.optimize()
    result = solution(model, variables)
    assert result.status == "optimal"
    assert math.isclose(result.objective, 20 / 3, abs_tol=1e-7)
    assert all(math.isclose(value, 4 / 3, abs_tol=1e-7) for value in result.values)
    # SCIP internally minimizes, so maximization LP duals need a sign change.
    duals = tuple(-model.getDualsolLinear(cons) for cons in constraints)
    assert all(
        math.isclose(value, expected, abs_tol=1e-7)
        for value, expected in zip(duals, (4 / 3, 1 / 3))
    )
    print("LP:", result, "max-sense duals:", duals)

    model.freeTransform()
    model.chgRhs(constraints[0], 3)
    model.optimize()
    modified = solution(model, variables)
    assert modified.status == "optimal"
    assert math.isclose(modified.objective, 16 / 3, abs_tol=1e-7)
    print("LP after RHS edit:", modified)

    model, variables = build_milp()
    for relax, expected in ((False, 47.0), (True, 47.55), (False, 47.0)):
        if model.getStage() > pyscipopt.SCIP_STAGE.PROBLEM:
            model.freeTransform()
        model.chgVarType(variables[1], "C" if relax else "I")
        model.chgVarType(variables[2], "C" if relax else "B")
        model.optimize()
        result = solution(model, variables)
        assert result.status == "optimal"
        assert math.isclose(result.objective, expected, abs_tol=1e-7)
        print("MILP relaxation:" if relax else "MILP:", result)

    model, variables, _ = build_knapsack()
    model.optimize()
    result = solution(model, variables)
    assert result.status == "optimal"
    assert math.isclose(result.objective, 41, abs_tol=1e-7)
    print("Knapsack:", result)

    model, variables = build_queens()
    model.optimize()
    result = solution(model, variables)
    assert result.status == "optimal"
    assert math.isclose(sum(result.values), 8, abs_tol=1e-7)
    print("8-queens: optimal, 8 queens")


def ring_rows(size: int, width: int) -> tuple[tuple[int, ...], ...]:
    """Even ring windows have the known LP and binary optimum size / 2."""
    if size < 2 or size % 2 or width < 2 or width % 2 or width > size:
        raise ValueError(
            "size and width must be positive even numbers, 2 <= width <= size"
        )
    return tuple(tuple((i + j) % size for j in range(width)) for i in range(size))


def check_ring_solution(size, rows, values, objective, *, binary, optimal) -> None:
    assert objective is not None
    assert len(values) == size
    assert all(value is not None and -1e-6 <= value <= 1 + 1e-6 for value in values)
    assert all(
        sum(values[j] for j in indices) <= len(indices) / 2 + 1e-6 for indices in rows
    )
    assert math.isclose(objective, sum(values), abs_tol=1e-6)
    if binary:
        assert all(abs(value - round(value)) <= 1e-6 for value in values)
    if optimal:
        assert math.isclose(objective, size / 2, abs_tol=1e-6)


def build_native(
    size, rows, strategy, *, binary=False, solve=False, presolve=True
) -> BuildTiming:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown SCIP building strategy: {strategy}")
    start = perf_counter()
    model = new_model("build")
    if not presolve:
        model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setMaximize()
    setup_done = perf_counter()
    variables = [
        model.addVar(f"x{i}", vtype="B" if binary else "C", ub=1, obj=1)
        for i in range(size)
    ]
    terms = [Term(var) for var in variables] if strategy in ("terms", "addconss") else []
    variables_done = perf_counter()
    pending = []
    for i, indices in enumerate(rows):
        rhs = ROW_COEFFICIENT * len(indices) / 2
        name = f"r{i}"
        if strategy == "coefficients":
            cons = model.addCons(Expr() <= rhs, name=name)
            for index in indices:
                model.addCoefLinear(cons, variables[index], ROW_COEFFICIENT)
        elif strategy == "quicksum":
            model.addCons(
                quicksum(ROW_COEFFICIENT * variables[index] for index in indices) <= rhs,
                name=name,
            )
        else:
            expr = Expr({terms[index]: ROW_COEFFICIENT for index in indices}) <= rhs
            if strategy == "addconss":
                pending.append(expr)
            else:
                model.addCons(expr, name=name)
    constraints_done = perf_counter()
    if strategy == "addconss":
        model.addConss(pending, name=[f"r{i}" for i in range(len(rows))])
    flush_done = perf_counter()
    assert model.getNVars() == size
    assert model.getNConss() == len(rows)
    status = "not-solved"
    solve_time = 0.0
    if solve:
        solve_start = perf_counter()
        model.optimize()
        solve_time = perf_counter() - solve_start
        result = solution(model, variables)
        status = result.status
        check_ring_solution(
            size,
            rows,
            result.values,
            result.objective,
            binary=binary,
            optimal=status == "optimal",
        )
    return BuildTiming(
        setup_done - start,
        variables_done - setup_done,
        constraints_done - variables_done,
        flush_done - constraints_done,
        solve_time,
        status,
    )


def build_python_mip(
    size, rows, solver, *, binary=False, solve=False, threads=1
) -> BuildTiming:
    import mip

    start = perf_counter()
    model = mip.Model(solver_name=solver, sense=mip.MAXIMIZE)
    model.verbose = 0
    model.threads = threads
    setup_done = perf_counter()
    variables = [
        model.add_var(
            f"x{i}", var_type=mip.BINARY if binary else mip.CONTINUOUS, ub=1, obj=1
        )
        for i in range(size)
    ]
    variables_done = perf_counter()
    for i, indices in enumerate(rows):
        model.add_constr(
            mip.xsum(ROW_COEFFICIENT * variables[j] for j in indices)
            <= ROW_COEFFICIENT * len(indices) / 2,
            f"r{i}",
        )
    constraints_done = perf_counter()
    # These backend-specific flushes are intentionally part of build time.
    if solver.upper() == mip.HIGHS.upper():
        model.solver._flush()
    elif solver.upper() == mip.CBC:
        import mip.cbc

        # CBC's row query commits its native column/row cache.
        mip.cbc.cbclib.Cbc_getRowNz(model.solver._model, 0)
    elif solver.upper() != mip.SCIP:
        raise ValueError(f"No measured flush path for solver {solver}")
    flush_done = perf_counter()
    assert model.num_cols == size
    assert model.num_rows == len(rows)
    status = "not-solved"
    solve_time = 0.0
    if solve:
        solve_start = perf_counter()
        status = model.optimize(max_seconds=10).name
        solve_time = perf_counter() - solve_start
        check_ring_solution(
            size,
            rows,
            tuple(var.x for var in variables),
            model.objective_value,
            binary=binary,
            optimal=model.status == mip.OptimizationStatus.OPTIMAL,
        )
    return BuildTiming(
        setup_done - start,
        variables_done - setup_done,
        constraints_done - variables_done,
        flush_done - constraints_done,
        solve_time,
        status,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000])
    parser.add_argument("--widths", nargs="+", type=int, default=[8, 128])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--binary", action="store_true")
    parser.add_argument(
        "--scip-no-presolve",
        action="store_true",
        help="Disable presolve in native SCIP paths only, for LP policy comparisons",
    )
    parser.add_argument(
        "--python-mip", nargs="*", choices=("CBC", "HIGHS", "SCIP"), default=[]
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    version_model = new_model("version")
    print(
        f"Python {platform.python_version()}, {platform.machine()}, "
        f"PySCIPOpt {pyscipopt.__version__}, "
        f"SCIP {version_model.getMajorVersion()}."
        f"{version_model.getMinorVersion()}.{version_model.getTechVersion()}"
    )
    del version_model
    check_small_models()
    if args.checks_only:
        return
    print(
        "Median seconds; setup/variables/rows/flush measured separately; total includes flush."
    )
    print(
        "shape             method               setup    vars    rows   flush   total   solve status"
    )
    for size in args.sizes:
        for width in args.widths:
            rows = ring_rows(size, width)
            for method in (*STRATEGIES, *args.python_mip):
                timings = [
                    (
                        build_native(
                            size,
                            rows,
                            method,
                            binary=args.binary,
                            solve=args.solve,
                            presolve=not args.scip_no_presolve,
                        )
                        if method in STRATEGIES
                        else build_python_mip(
                            size, rows, method, binary=args.binary, solve=args.solve
                        )
                    )
                    for _ in range(args.repeats)
                ]
                label = f"SCIP/{method}" if method in STRATEGIES else f"mip/{method}"
                values = [
                    median(getattr(t, field) for t in timings)
                    for field in (
                        "setup",
                        "variables",
                        "constraints",
                        "flush",
                        "total",
                        "solve",
                    )
                ]
                print(
                    f"{size}x{size} w={width:<4} {label:<20} "
                    + " ".join(f"{value:7.3f}" for value in values)
                    + " "
                    + ",".join(sorted({t.status for t in timings})),
                    flush=True,
                )


if __name__ == "__main__":
    main()
