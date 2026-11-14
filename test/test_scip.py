"""SCIP-specific regressions for the public Python-MIP API."""

import pytest

scip = pytest.importorskip("pyscipopt")

import mip


def model(**kwargs):
    result = mip.Model(solver_name=mip.SCIP, **kwargs)
    result.verbose = 0
    return result


@pytest.mark.parametrize("sense", (mip.MINIMIZE, mip.MAXIMIZE))
def test_repeated_lp_duals_and_redundant_edit(sense):
    m = model(sense=sense)
    sign = 1 if sense == mip.MAXIMIZE else -1
    x = m.add_var(obj=sign * 3)
    y = m.add_var(obj=sign * 2)
    capacity = m.add_constr(2 * x + y <= 4)
    balance = m.add_constr(x + 2 * y <= 4)
    for edit in (False, False, True):
        if edit:
            x.ub = 10
            m += x + y <= 10
        assert m.optimize() == mip.OptimizationStatus.OPTIMAL
        assert m.objective_value == pytest.approx(sign * 20 / 3)
        assert (capacity.pi, balance.pi) == pytest.approx((sign * 4 / 3, sign / 3))
        assert (x.rc, y.rc) == pytest.approx((0, 0))
    m.check_optimization_results()


def test_singleton_duals_and_column_after_solve():
    m = model()
    x = m.add_var(obj=1)
    demand = m.add_constr(x >= 2)
    assert m.optimize() == mip.OptimizationStatus.OPTIMAL
    assert demand.pi == pytest.approx(1)
    y = m.add_var(obj=0.4, column=mip.Column([demand], [1]))
    assert x.x is None
    assert m.optimize() == mip.OptimizationStatus.OPTIMAL
    assert (x.x, y.x) == pytest.approx((0, 2))
    assert demand.pi == pytest.approx(0.4)
    assert x.column.coeffs == [1]
    assert demand.expr.expr == pytest.approx({x: 1, y: 1})


def test_relaxation_preserves_fractional_bounds_and_solution():
    m = model(sense=mip.MAXIMIZE)
    x = m.add_var(var_type=mip.INTEGER, lb=0.25, ub=0.75, obj=1)
    assert m.optimize() == mip.OptimizationStatus.INFEASIBLE
    assert m.optimize(relax=True) == mip.OptimizationStatus.OPTIMAL
    assert x.x == pytest.approx(0.75)
    assert x.var_type == mip.INTEGER
    assert (x.lb, x.ub) == (0.25, 0.75)
    assert m.objective_value == pytest.approx(0.75)
    assert m.objective_bound == pytest.approx(0.75)
    assert m.optimize() == mip.OptimizationStatus.INFEASIBLE
    assert x.x is None


def test_lp_after_mip_restores_original_bounds_and_row_duals():
    m = model(sense=mip.MAXIMIZE)
    x = m.add_var(var_type=mip.INTEGER, ub=5, obj=1)
    limit = m.add_constr(x <= 4)
    assert m.optimize() == mip.OptimizationStatus.OPTIMAL
    assert m.optimize(relax=True) == mip.OptimizationStatus.OPTIMAL
    assert x.x == pytest.approx(4)
    assert x.ub == 5
    assert limit.pi == pytest.approx(1)
    assert x.rc == pytest.approx(0)
    assert m.optimize() == mip.OptimizationStatus.OPTIMAL
    assert x.rc is None
    assert limit.pi is None


def test_removal_reindexes_rows_columns_and_names():
    m = model()
    x, y, z = [m.add_var(name=name, obj=1) for name in ("x", "y", "z")]
    deleted = m.add_constr(x + y >= 1, name="deleted")
    kept = m.add_constr(y + 2 * z >= 4, name="kept")
    m.remove(deleted)
    m.remove(y)
    assert m.var_by_name("y") is None
    assert m.constr_by_name("deleted") is None
    assert m.var_by_name("z") is z
    assert kept.idx == 0
    assert z.idx == 1
    assert kept.expr.expr == pytest.approx({z: 2})
    assert m.num_nz == 1
    assert m.optimize() == mip.OptimizationStatus.OPTIMAL
    assert (x.x, z.x) == pytest.approx((0, 2))


def test_constraint_replacement_and_rhs_change():
    m = model(sense=mip.MAXIMIZE)
    x = m.add_var(ub=10, obj=1)
    c = m.add_constr(x <= 3, name="limit")
    assert m.optimize() == mip.OptimizationStatus.OPTIMAL
    c.expr = 2 * x <= 8
    c.rhs = 6
    assert m.optimize() == mip.OptimizationStatus.OPTIMAL
    assert x.x == pytest.approx(3)
    assert c.pi == pytest.approx(0.5)
    assert m.constr_by_name("limit") is c


@pytest.mark.parametrize("integer", (False, True))
def test_zero_time_limit_has_no_stale_primal(integer):
    m = model(sense=mip.MAXIMIZE)
    x = m.add_var(ub=10, var_type=mip.INTEGER if integer else mip.CONTINUOUS, obj=1)
    assert m.optimize() == mip.OptimizationStatus.OPTIMAL
    m.reset()
    status = m.optimize(max_seconds=0)
    assert status in (
        mip.OptimizationStatus.NO_SOLUTION_FOUND,
        mip.OptimizationStatus.FEASIBLE,
        mip.OptimizationStatus.TRUNCATED,
    )
    if m.num_solutions == 0:
        assert x.x is None
        assert m.objective_value is None


def test_lp_iteration_limit_stops_without_retrying_and_preserves_node_limit():
    m = model(sense=mip.MAXIMIZE)
    x = m.add_var(obj=3)
    y = m.add_var(obj=2)
    m += 2 * x + y <= 4
    m += x + 2 * y <= 4
    m.max_nodes = 3
    m.max_iter = 1
    assert m.optimize(max_seconds=1) == mip.OptimizationStatus.TRUNCATED
    assert m.solver._scip.getStatus() == "nodelimit"
    assert m.solver._scip.getNLPIterations() == 1
    assert m.max_nodes == 3
    m.max_iter = mip.INT_MAX
    assert m.optimize() == mip.OptimizationStatus.OPTIMAL
    assert m.objective_value == pytest.approx(20 / 3)
    assert (x.x, y.x) == pytest.approx((4 / 3, 4 / 3))
    m.check_optimization_results()


def test_clear_preserves_solver_sense_and_verbosity():
    m = model(sense=mip.MAXIMIZE)
    m.add_var(obj=1, ub=2)
    m.optimize()
    m.clear()
    assert m.solver_name == mip.SCIP
    assert m.sense == mip.MAXIMIZE
    assert m.verbose == 0
    assert m.num_cols == m.num_rows == m.num_solutions == 0
    x = m.add_var(obj=1, ub=3)
    assert m.optimize() == mip.OptimizationStatus.OPTIMAL
    assert x.x == pytest.approx(3)


def test_write_preserves_cached_solution(tmp_path):
    m = model()
    x = m.add_var(obj=1)
    m += x >= 2
    assert m.optimize() == mip.OptimizationStatus.OPTIMAL
    m.write(str(tmp_path / "model.lp"))
    assert x.x == pytest.approx(2)
    assert m.objective_value == pytest.approx(2)
    assert m.status == mip.OptimizationStatus.OPTIMAL


@pytest.mark.parametrize("extension", ("lp", "mps"))
@pytest.mark.parametrize("sense", (mip.MINIMIZE, mip.MAXIMIZE))
def test_linear_model_round_trip(tmp_path, extension, sense):
    m = model(sense=sense)
    x = m.add_var(name="x", var_type=mip.BINARY)
    y = m.add_var(name="y", ub=3)
    m += 2 * x + y >= 1.5, "demand"
    m.objective = x + 2 * y + 0.5
    assert m.optimize() == mip.OptimizationStatus.OPTIMAL
    expected = m.objective_value
    path = str(tmp_path / f"model.{extension}")
    m.write(path)
    restored = model()
    restored.read(path)
    assert restored.num_cols == 2
    assert restored.num_rows == 1
    assert restored.num_int == 1
    assert restored.objective_const == pytest.approx(0.5)
    assert restored.sense == sense
    assert restored.var_by_name("x").var_type == mip.BINARY
    assert restored.optimize() == mip.OptimizationStatus.OPTIMAL
    assert restored.objective_value == pytest.approx(expected)
    restored.check_optimization_results()


def test_ranged_mps_row_is_preserved(tmp_path):
    path = tmp_path / "ranged.mps"
    path.write_text(
        "NAME          RANGED\n"
        "ROWS\n"
        " N  OBJ\n"
        " L  BAND\n"
        "COLUMNS\n"
        "    X         OBJ       1             BAND      1\n"
        "RHS\n"
        "    RHS1      BAND      3\n"
        "RANGES\n"
        "    RNG1      BAND      2\n"
        "BOUNDS\n"
        " UP BND1      X         5\n"
        "ENDATA\n"
    )
    m = model()
    m.read(str(path))
    assert m.num_rows == 2
    assert m.optimize() == mip.OptimizationStatus.OPTIMAL
    assert m.objective_value == pytest.approx(1)
    m.sense = mip.MAXIMIZE
    assert m.optimize() == mip.OptimizationStatus.OPTIMAL
    assert m.objective_value == pytest.approx(3)


def test_partial_mip_start_after_lp_relaxation():
    m = model(sense=mip.MAXIMIZE)
    x = m.add_var(var_type=mip.BINARY, obj=1)
    y = m.add_var(var_type=mip.BINARY, obj=2)
    m += x + y <= 1
    assert m.optimize(relax=True) == mip.OptimizationStatus.OPTIMAL
    m.start = [(y, 1)]
    assert m.optimize() == mip.OptimizationStatus.OPTIMAL
    assert (x.x, y.x) == pytest.approx((0, 1))


def test_sparse_mip_start_before_node_search():
    m = model()
    m.preprocess = 0
    m.solver._scip.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
    m.solver._scip.setIntParam("heuristics/completesol/freq", 0)
    x = [m.add_var(var_type=mip.BINARY, obj=3 if i == 0 else 1) for i in range(20)]
    m += mip.xsum(x) >= 2
    m.start = [(x[0], 1)]
    assert m.optimize(max_nodes=0) == mip.OptimizationStatus.FEASIBLE
    assert m.solver._scip.getNNodes() == 0
    assert m.objective_value == pytest.approx(4)
    assert x[0].x == pytest.approx(1)
    m.check_optimization_results()


@pytest.mark.parametrize("relax_first", (False, True))
def test_complete_mip_start_before_node_search(relax_first):
    m = model(sense=mip.MAXIMIZE)
    m.preprocess = 0
    m.solver._scip.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
    x = m.add_var(var_type=mip.BINARY, obj=1)
    y = m.add_var(var_type=mip.BINARY, obj=2)
    m += x + y <= 1
    if relax_first:
        assert m.optimize(relax=True) == mip.OptimizationStatus.OPTIMAL
    m.start = [(x, 0), (y, 1)]
    assert m.optimize(max_nodes=0) == mip.OptimizationStatus.FEASIBLE
    assert m.solver._scip.getNNodes() == 0
    assert m.objective_value == pytest.approx(2)
    assert (x.x, y.x) == pytest.approx((0, 1))
    m.check_optimization_results()


def test_round_trip_keeps_original_integer_lp_bounds(tmp_path):
    m = model(sense=mip.MAXIMIZE)
    x = m.add_var(name="x", var_type=mip.INTEGER, lb=0.25, ub=0.75, obj=1)
    assert m.optimize(relax=True) == mip.OptimizationStatus.OPTIMAL
    path = str(tmp_path / "fractional.lp")
    m.write(path)
    assert x.x == pytest.approx(0.75)
    restored = model()
    restored.read(path)
    assert restored.optimize() == mip.OptimizationStatus.INFEASIBLE
    assert restored.optimize(relax=True) == mip.OptimizationStatus.OPTIMAL
    assert restored.objective_value == pytest.approx(0.75)


def test_invalid_column_does_not_modify_model():
    m = model()
    x = m.add_var()
    c = m.add_constr(x >= 1)
    with pytest.raises(ValueError, match="must match"):
        m.add_var(column=mip.Column([c], []))
    other = model()
    y = other.add_var()
    foreign = other.add_constr(y >= 1)
    with pytest.raises(ValueError, match="belong"):
        m.add_var(column=mip.Column([foreign], [1]))
    assert m.num_cols == 1


@pytest.mark.parametrize("callback", ("cuts_generator", "lazy_constrs_generator"))
def test_unsupported_callbacks_fail_before_solving(callback):
    m = model()
    m.add_var()
    setattr(m, callback, mip.ConstrsGenerator())
    with pytest.raises(NotImplementedError, match="callbacks"):
        m.optimize()
    assert m.num_solutions == 0


def test_solution_pool_indices_match_objectives():
    m = model(sense=mip.MAXIMIZE)
    x = m.add_var(var_type=mip.BINARY, obj=1)
    y = m.add_var(var_type=mip.BINARY, obj=2)
    m += x + y <= 1
    assert m.optimize() == mip.OptimizationStatus.OPTIMAL
    assert m.objective_values == sorted(m.objective_values, reverse=True)
    for i, objective in enumerate(m.objective_values):
        assert x.xi(i) + 2 * y.xi(i) == pytest.approx(objective)
