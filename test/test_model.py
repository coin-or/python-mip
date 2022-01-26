import os

import pytest

from mip import (
    CBC,
    GUROBI,
    Model,
    MAXIMIZE,
    MINIMIZE,
    OptimizationStatus,
    INTEGER,
    CONTINUOUS,
    BINARY,
)

TOL = 1e-4
SOLVERS = [CBC]
if "GUROBI_HOME" in os.environ:
    SOLVERS += [GUROBI]

# Overall Optimization Tests


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("var_type", (CONTINUOUS, INTEGER))
@pytest.mark.parametrize(
    "sense,status,xvalue,obj_value",
    (
        (MAXIMIZE, OptimizationStatus.UNBOUNDED, None, None),  # unbounded case
        (MINIMIZE, OptimizationStatus.OPTIMAL, 0, 0),  # implicit lower bound 0
    ),
)
def test_single_continuous_or_integer_variable_with_default_bounds(
    solver, var_type, sense: str, status, xvalue, obj_value
):
    m = Model(solver_name=solver, sense=sense)
    x = m.add_var(name="x", var_type=var_type, obj=1)
    m.optimize()
    # check result
    assert m.status == status
    assert x.x == xvalue
    assert m.objective_value == obj_value


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize(
    "sense,status,xvalue,objvalue",
    [
        (MAXIMIZE, OptimizationStatus.OPTIMAL, 1, 1),  # implicit upper bound 1
        (MINIMIZE, OptimizationStatus.OPTIMAL, 0, 0),  # implicit lower bound 0
    ],
)
def test_single_binary_variable_with_default_bounds(
    solver, sense: str, status, xvalue, objvalue
):
    m = Model(solver_name=solver, sense=sense)
    x = m.add_var(name="x", var_type=BINARY, obj=1)
    m.optimize()
    # check result
    assert m.status == status
    assert x.x == xvalue
    assert m.objective_value == objvalue


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("var_type", (CONTINUOUS, INTEGER))
@pytest.mark.parametrize(
    "lb,ub,min_obj,max_obj",
    (
        (0, 0, 0, 0),  # fixed to 0
        (2, 2, 2, 2),  # fixed to positive
        (-2, -2, -2, -2),  # fixed to negative
        (1, 2, 1, 2),  # positive range
        (-3, 2, -3, 2),  # negative range
        (-4, 5, -4, 5),  # range from positive to negative
    ),
)
def test_single_continuous_or_integer_variable_with_different_bounds(
    solver, var_type, lb, ub, min_obj, max_obj
):
    # Minimum Case
    m = Model(solver_name=solver, sense=MINIMIZE)
    m.add_var(name="x", var_type=var_type, lb=lb, ub=ub, obj=1)
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert m.objective_value == min_obj

    # Maximum Case
    m = Model(solver_name=solver, sense=MAXIMIZE)
    m.add_var(name="x", var_type=var_type, lb=lb, ub=ub, obj=1)
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert m.objective_value == max_obj


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize(
    "lb,ub,min_obj,max_obj",
    (
        (0, 1, 0, 1),  # regular case
        (0, 0, 0, 0),  # fixed to 0
        (1, 1, 1, 1),  # fixed to 1
    ),
)
def test_binary_variable_with_different_bounds(solver, lb, ub, min_obj, max_obj):
    # Minimum Case
    m = Model(solver_name=solver, sense=MINIMIZE)
    m.add_var(name="x", var_type=BINARY, lb=lb, ub=ub, obj=1)
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert m.objective_value == min_obj

    # Maximum Case
    m = Model(solver_name=solver, sense=MAXIMIZE)
    m.add_var(name="x", var_type=BINARY, lb=lb, ub=ub, obj=1)
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert m.objective_value == max_obj


@pytest.mark.parametrize("solver", SOLVERS)
def test_binary_variable_illegal_bounds(solver):
    m = Model(solver_name=solver)
    # Illegal lower bound
    with pytest.raises(ValueError):
        m.add_var("x", lb=-1, var_type=BINARY)
    # Illegal upper bound
    with pytest.raises(ValueError):
        m.add_var("x", ub=2, var_type=BINARY)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("sense", (MINIMIZE, MAXIMIZE))
@pytest.mark.parametrize(
    "var_type,lb,ub",
    (
        (CONTINUOUS, 3.5, 2),
        (INTEGER, 5, 4),
        (BINARY, 1, 0),
    ),
)
def test_contradictory_variable_bounds(solver, sense: str, var_type: str, lb, ub):
    m = Model(solver_name=solver, sense=sense)
    m.add_var(name="x", var_type=var_type, lb=lb, ub=ub, obj=1)
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.INFEASIBLE


@pytest.mark.parametrize("solver", SOLVERS)
def test_float_bounds_for_integer_variable(solver):
    # Minimum Case
    m = Model(solver_name=solver, sense=MINIMIZE)
    m.add_var(name="x", var_type=INTEGER, lb=-1.5, ub=3.5, obj=1)
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert m.objective_value == -1

    # Maximum Case
    m = Model(solver_name=solver, sense=MAXIMIZE)
    m.add_var(name="x", var_type=INTEGER, lb=-1.5, ub=3.5, obj=1)
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert m.objective_value == 3


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("sense", (MINIMIZE, MAXIMIZE))
def test_single_default_variable_with_nothing_to_do(solver, sense):
    m = Model(solver_name=solver, sense=sense)
    m.add_var(name="x")
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert m.objective_value == 0


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("var_type", (CONTINUOUS, INTEGER, BINARY))
@pytest.mark.parametrize("obj", (1.2, 2))
def test_single_variable_with_different_non_zero_objectives(solver, var_type, obj):
    # Maximize
    m = Model(solver_name=solver, sense=MAXIMIZE)
    x = m.add_var(name="x", var_type=var_type, lb=0, ub=1, obj=obj)
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert m.objective_value == obj
    assert x.x == 1.0
    # Minimize with negative
    m = Model(solver_name=solver, sense=MINIMIZE)
    x = m.add_var(name="x", var_type=var_type, lb=0, ub=1, obj=-obj)
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert m.objective_value == -obj
    assert x.x == 1.0
