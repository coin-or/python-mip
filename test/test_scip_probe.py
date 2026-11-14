"""Native SCIP probes for the first stage of backend development."""

import pytest

pytest.importorskip("pyscipopt")

from benchmarks.scip_probe import (
    STRATEGIES,
    build_knapsack,
    build_lp,
    build_milp,
    build_native,
    build_python_mip,
    build_queens,
    new_model,
    ring_rows,
    solution,
)


@pytest.mark.parametrize("maximize", (False, True))
def test_lp_primal_dual_and_reduced_cost(maximize):
    model, variables, constraints = build_lp(maximize=maximize)
    model.optimize()
    result = solution(model, variables)
    sign = 1 if maximize else -1
    assert result.status == "optimal"
    assert result.values == pytest.approx((4 / 3, 4 / 3))
    assert result.objective == pytest.approx(sign * 20 / 3)
    assert result.bound == pytest.approx(result.objective)
    # Both models internally minimize -3*x - 2*y.
    assert tuple(model.getDualsolLinear(c) for c in constraints) == pytest.approx(
        (-4 / 3, -1 / 3)
    )
    assert tuple(model.getVarRedcost(v) for v in variables) == pytest.approx((0, 0))


def test_lp_modification_after_solve():
    model, variables, constraints = build_lp()
    model.optimize()
    original = solution(model, variables)
    model.freeTransform()
    model.chgRhs(constraints[0], 3)
    model.optimize()
    modified = solution(model, variables)
    assert original.objective == pytest.approx(20 / 3)
    assert modified.status == "optimal"
    assert modified.objective == pytest.approx(16 / 3)
    assert modified.values == pytest.approx((2 / 3, 5 / 3))


def test_milp_relaxation_and_restore():
    model, variables = build_milp()
    model.optimize()
    integer = solution(model, variables)
    assert integer.status == "optimal"
    assert integer.objective == pytest.approx(47)
    assert integer.values == pytest.approx((9.5, 10, 1))

    model.freeTransform()
    model.chgVarType(variables[1], "C")
    model.chgVarType(variables[2], "C")
    model.optimize()
    relaxed = solution(model, variables)
    assert relaxed.status == "optimal"
    assert relaxed.objective == pytest.approx(47.55)
    assert relaxed.values == pytest.approx((9.5, 10.5, 0.95))

    model.freeTransform()
    model.chgVarType(variables[1], "I")
    model.chgVarType(variables[2], "B")
    model.optimize()
    restored = solution(model, variables)
    assert restored.status == "optimal"
    assert restored.objective == pytest.approx(integer.objective)
    assert restored.values == pytest.approx(integer.values)
    assert relaxed.objective == pytest.approx(47.55)  # copied before restoring types


def test_knapsack_matches_existing_example():
    model, variables, constraint = build_knapsack()
    model.optimize()
    result = solution(model, variables)
    assert result.status == "optimal"
    assert result.objective == pytest.approx(41)
    assert all(abs(value - round(value)) < 1e-7 for value in result.values)
    assert model.getSlack(constraint, sol=model.getBestSol()) == pytest.approx(1)


def test_queens_feasibility():
    n = 8
    model, variables = build_queens(n)
    model.optimize()
    result = solution(model, variables)
    assert result.status == "optimal"
    selected = [(i // n, i % n) for i, value in enumerate(result.values) if value > 0.99]
    assert len(selected) == n
    assert len({i for i, _ in selected}) == n
    assert len({j for _, j in selected}) == n
    assert len({i - j for i, j in selected}) == n
    assert len({i + j for i, j in selected}) == n


@pytest.mark.parametrize("kind", ("infeasible", "unbounded"))
@pytest.mark.parametrize("integer", (False, True))
def test_no_usable_solution_for_infeasible_or_unbounded(kind, integer):
    model = new_model(kind)
    x = model.addVar("x", vtype="I" if integer else "C", obj=1)
    if kind == "infeasible":
        model.addCons(x >= 2)
        model.addCons(x <= 1)
    else:
        model.setMaximize()
    model.optimize()
    result = solution(model, (x,))
    assert result.status == kind
    assert result.objective is None
    assert result.values == ()


def test_lp_column_insertion():
    model = new_model("master", lp_duals=True)
    x = model.addVar("x", obj=1)
    demand = model.addCons(x >= 2, modifiable=True)
    model.optimize()
    assert solution(model, (x,)).objective == pytest.approx(2)
    assert model.getDualsolLinear(demand) == pytest.approx(1)

    model.freeTransform()
    y = model.addVar("new_column", obj=0.4)
    model.addCoefLinear(demand, y, 1)
    model.optimize()
    result = solution(model, (x, y))
    assert result.status == "optimal"
    assert result.objective == pytest.approx(0.8)
    assert result.values == pytest.approx((0, 2))
    assert model.getDualsolLinear(demand) == pytest.approx(0.4)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("binary", (False, True))
def test_build_strategies_solve_identical_models(strategy, binary):
    timing = build_native(16, ring_rows(16, 8), strategy, binary=binary, solve=True)
    assert timing.status == "optimal"


def test_lp_benchmark_without_presolve():
    timing = build_native(
        16, ring_rows(16, 8), "coefficients", solve=True, presolve=False
    )
    assert timing.status == "optimal"


@pytest.mark.parametrize("solver", ("CBC", "HIGHS", "SCIP"))
@pytest.mark.parametrize("binary", (False, True))
def test_python_mip_benchmark_baselines(solver, binary):
    if solver == "HIGHS":
        from mip.highs import has_highs

        if not has_highs:
            pytest.skip("HiGHS not available")
    # HiGHS shares its scheduler across models; do not reconfigure a scheduler
    # already initialized by another test. The standalone benchmark uses one thread.
    timing = build_python_mip(
        16, ring_rows(16, 8), solver, binary=binary, solve=True, threads=0
    )
    assert timing.status == "OPTIMAL"
