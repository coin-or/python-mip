import re
from os import environ

import pytest

import mip
import mip.gurobi
import mip.highs
from mip import (
    CBC,
    Column,
    GUROBI,
    HIGHS,
    LinExpr,
    Model,
    MAXIMIZE,
    MINIMIZE,
    OptimizationStatus,
    Var,
    INTEGER,
    CONTINUOUS,
    BINARY,
)
from util import skip_on

TOL = 1e-4
SOLVERS = [CBC]
if mip.gurobi.has_gurobi and "GUROBI_HOME" in environ:
    SOLVERS += [GUROBI]
if mip.highs.has_highs:
    SOLVERS += [HIGHS]

# Overall Optimization Tests


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("var_type", (CONTINUOUS, INTEGER))
def test_minimize_single_continuous_or_integer_variable_with_default_bounds(
    solver, var_type
):
    m = Model(solver_name=solver, sense=MINIMIZE)
    x = m.add_var(name="x", var_type=var_type, obj=1)
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert abs(x.x) < TOL
    assert abs(m.objective_value) < TOL


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("var_type", (CONTINUOUS, INTEGER))
def test_maximize_single_continuous_or_integer_variable_with_default_bounds(
    solver, var_type
):
    m = Model(solver_name=solver, sense=MAXIMIZE)
    x = m.add_var(name="x", var_type=var_type, obj=1)
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.UNBOUNDED
    assert x.x is None
    assert m.objective_value is None


@skip_on(NotImplementedError)
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
    assert abs(x.x - xvalue) < TOL
    assert abs(m.objective_value - objvalue) < TOL


@skip_on(NotImplementedError)
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
    assert abs(m.objective_value - min_obj) < TOL

    # Maximum Case
    m = Model(solver_name=solver, sense=MAXIMIZE)
    m.add_var(name="x", var_type=var_type, lb=lb, ub=ub, obj=1)
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - max_obj) < TOL


@skip_on(NotImplementedError)
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
    assert abs(m.objective_value - min_obj) < TOL

    # Maximum Case
    m = Model(solver_name=solver, sense=MAXIMIZE)
    m.add_var(name="x", var_type=BINARY, lb=lb, ub=ub, obj=1)
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - max_obj) < TOL


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_binary_variable_illegal_bounds(solver):
    m = Model(solver_name=solver)
    # Illegal lower bound
    with pytest.raises(ValueError):
        m.add_var("x", lb=-1, var_type=BINARY)
    # Illegal upper bound
    with pytest.raises(ValueError):
        m.add_var("x", ub=2, var_type=BINARY)


@skip_on(NotImplementedError)
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


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_float_bounds_for_integer_variable(solver):
    # Minimum Case
    m = Model(solver_name=solver, sense=MINIMIZE)
    m.add_var(name="x", var_type=INTEGER, lb=-1.5, ub=3.5, obj=1)
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - (-1)) < TOL

    # Maximum Case
    m = Model(solver_name=solver, sense=MAXIMIZE)
    m.add_var(name="x", var_type=INTEGER, lb=-1.5, ub=3.5, obj=1)
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - 3) < TOL


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("sense", (MINIMIZE, MAXIMIZE))
def test_single_default_variable_with_nothing_to_do(solver, sense):
    m = Model(solver_name=solver, sense=sense)
    m.add_var(name="x")
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value) < TOL


@skip_on(NotImplementedError)
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
    assert abs(m.objective_value - obj) < TOL
    assert abs(x.x - 1.0) < TOL
    # Minimize with negative
    m = Model(solver_name=solver, sense=MINIMIZE)
    x = m.add_var(name="x", var_type=var_type, lb=0, ub=1, obj=-obj)
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - (-obj)) < TOL
    assert abs(x.x - 1.0) < TOL


# Variable Tests

@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_hashes_of_variables(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")

    assert hash(x) == 0
    assert hash(y) == 1


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("constant", (-1, 1.2, 2))
def test_addition_of_var_with_non_zero_constant(solver, constant):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    y_right = x + constant
    assert isinstance(y_right, LinExpr)
    assert y_right.const == constant
    assert y_right.expr == {x: 1}

    y_left = constant + x
    assert isinstance(y_left, LinExpr)
    assert y_left.const == constant
    assert y_left.expr == {x: 1}

    # in-place
    y = x
    y += constant
    assert isinstance(y, LinExpr)
    assert y.const == constant
    assert y.expr == {x: 1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_addition_of_var_with_zero(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    y_right = x + 0
    assert isinstance(y_right, Var)
    assert hash(y_right) == hash(x)

    y_left = 0 + x
    assert isinstance(y_left, Var)
    assert hash(y_left) == hash(x)

    # in-place
    y = x
    y += 0
    assert isinstance(y, Var)
    assert hash(y) == hash(x)


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_addition_of_two_vars(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")

    z = x + y
    assert isinstance(z, LinExpr)
    assert z.const == 0
    assert z.expr == {x: 1, y: 1}

    # in-place
    z = x
    z += y
    assert isinstance(z, LinExpr)
    assert z.const == 0
    assert z.expr == {x: 1, y: 1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_addition_of_var_with_linear_expression(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    z = m.add_var(name="z")

    lin_expr = y + z

    l1 = x + lin_expr
    assert isinstance(l1, LinExpr)
    assert l1.const == 0
    assert l1.expr == {x: 1, y: 1, z: 1}

    # in-place
    w = x
    w += lin_expr
    assert isinstance(w, LinExpr)
    assert w.const == 0
    assert w.expr == {x: 1, y: 1, z: 1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_addition_of_var_with_illegal_type(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    with pytest.raises(TypeError):
        x + "1"

    with pytest.raises(TypeError):
        "1" + x


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("constant", (-1, 1.2, 2))
def test_subtraction_of_var_with_non_zero_constant(solver, constant):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    y_right = x - constant
    assert isinstance(y_right, LinExpr)
    assert y_right.const == -constant
    assert y_right.expr == {x: 1}

    y_left = constant - x
    assert isinstance(y_left, LinExpr)
    assert y_left.const == constant
    assert y_left.expr == {x: -1}

    # in-place
    y = x
    y -= constant
    assert isinstance(y, LinExpr)
    assert y.const == -constant
    assert y.expr == {x: 1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_subtraction_of_var_with_zero(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    y_right = x - 0
    assert isinstance(y_right, Var)
    assert hash(y_right) == hash(x)

    y_left = 0 - x
    assert isinstance(y_left, LinExpr)
    assert y_left.const == 0
    assert y_left.expr == {x: -1}

    # in-place
    y = x
    y -= 0
    assert isinstance(y, Var)
    assert hash(y) == hash(x)


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_subtraction_of_two_vars(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")

    z1 = x - y
    assert isinstance(z1, LinExpr)
    assert z1.const == 0
    assert z1.expr == {x: 1, y: -1}

    # in-place
    z = x
    z -= y
    assert isinstance(z, LinExpr)
    assert z.const == 0
    assert z.expr == {x: 1, y: -1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_subtraction_of_var_with_linear_expression(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    z = m.add_var(name="z")

    lin_expr = y + z

    l1 = x - lin_expr
    assert isinstance(l1, LinExpr)
    assert l1.const == 0
    assert l1.expr == {x: 1, y: -1, z: -1}

    # in-place
    w = x
    w -= lin_expr
    assert isinstance(w, LinExpr)
    assert w.const == 0
    assert w.expr == {x: 1, y: -1, z: -1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_subtraction_of_var_with_illegal_type(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    with pytest.raises(TypeError):
        x - "1"

    with pytest.raises(TypeError):
        "1" - x


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("coefficient", (-1, 0, 1.2, 2))
def test_multiply_var_with_coefficient(solver, coefficient):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    y_right = coefficient * x
    assert isinstance(y_right, LinExpr)
    assert y_right.const == 0.0
    assert y_right.expr == {x: coefficient}

    y_left = x * coefficient
    assert isinstance(y_left, LinExpr)
    assert y_left.const == 0.0
    assert y_left.expr == {x: coefficient}

    # in-place
    y = x
    y *= coefficient
    assert isinstance(y, LinExpr)
    assert y.const == 0.0
    assert y.expr == {x: coefficient}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_multiply_var_with_illegal_coefficient(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    with pytest.raises(TypeError):
        "1" * x

    with pytest.raises(TypeError):
        x * "1"


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("coefficient", (-1, 1.2, 2))
def test_divide_var_with_non_zero_coefficient(solver, coefficient):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    y = x / coefficient
    assert isinstance(y, LinExpr)
    assert y.const == 0.0
    assert y.expr == {x: 1/coefficient}

    # in-place
    y = x
    y /= coefficient
    assert isinstance(y, LinExpr)
    assert y.const == 0.0
    assert y.expr == {x: 1/coefficient}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_divide_var_with_illegal_coefficient(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    with pytest.raises(TypeError):
        y = x / "1"


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_divide_var_with_zero(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    with pytest.raises(ZeroDivisionError):
        x / 0


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_negate_variable(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    y = - x
    assert isinstance(y, LinExpr)
    assert y.const == 0.0
    assert y.expr == {x: -1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("constant", (-1, 0, 1))
def test_constraint_with_var_and_const(solver, constant):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    constr = x <= constant
    assert isinstance(constr, LinExpr)
    assert constr.const == -constant
    assert constr.expr == {x: 1}
    assert constr.sense == "<"

    constr = x == constant
    assert isinstance(constr, LinExpr)
    assert constr.const == -constant
    assert constr.expr == {x: 1}
    assert constr.sense == "="

    constr = x >= constant
    assert isinstance(constr, LinExpr)
    assert constr.const == -constant
    assert constr.expr == {x: 1}
    assert constr.sense == ">"

    # The following tests produce inverted sense for <= and >=,
    # as the constants operator doesn't support comparison with variables

    constr = constant <= x
    assert isinstance(constr, LinExpr)
    assert constr.const == -constant
    assert constr.expr == {x: 1}
    assert constr.sense == ">"

    constr = constant == x
    assert isinstance(constr, LinExpr)
    assert constr.const == -constant
    assert constr.expr == {x: 1}
    assert constr.sense == "="

    constr = constant >= x
    assert isinstance(constr, LinExpr)
    assert constr.const == -constant
    assert constr.expr == {x: 1}
    assert constr.sense == "<"


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_constraint_with_var_and_var(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")

    constr = x <= y
    assert isinstance(constr, LinExpr)
    assert constr.const == 0
    assert constr.expr == {x: 1, y: -1}
    assert constr.sense == "<"

    constr = x == y
    assert isinstance(constr, LinExpr)
    assert constr.const == 0
    assert constr.expr == {x: 1, y: -1}
    assert constr.sense == "="

    constr = x >= y
    assert isinstance(constr, LinExpr)
    assert constr.const == 0
    assert constr.expr == {x: 1, y: -1}
    assert constr.sense == ">"


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_constraint_with_var_and_lin_expr(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    z = m.add_var(name="z")

    term = y + z

    constr = x <= term
    assert isinstance(constr, LinExpr)
    assert constr.const == 0
    assert constr.expr == {x: 1, y: -1, z: -1}
    assert constr.sense == "<"

    constr = x == term
    assert isinstance(constr, LinExpr)
    assert constr.const == 0
    assert constr.expr == {x: 1, y: -1, z: -1}
    assert constr.sense == "="

    constr = x >= term
    assert isinstance(constr, LinExpr)
    assert constr.const == 0
    assert constr.expr == {x: 1, y: -1, z: -1}
    assert constr.sense == ">"


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_constraint_with_var_and_illegal_type(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    with pytest.raises(TypeError):
        x <= "1"

    with pytest.raises(TypeError):
        x == "1"

    with pytest.raises(TypeError):
        x >= "1"

    with pytest.raises(TypeError):
        "1" <= x

    with pytest.raises(TypeError):
        "1" == x

    with pytest.raises(TypeError):
        "1" >= x


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_query_variable_attributes(solver):
    m = Model(solver_name=solver, sense=MAXIMIZE)
    x = m.add_var(name="a_variable", lb=0.0, ub=5.0, obj=1.0)
    c = m.add_constr(1 <= x, "some_constraint")

    # Check before optimization
    assert x.name == "a_variable"
    assert str(x) == "a_variable"
    assert x.lb == 0.0
    assert x.ub == 5.0
    assert x.obj == 1.0
    assert x.var_type == CONTINUOUS
    assert x.branch_priority == 0
    column = x.column
    assert column.coeffs == [1]
    assert column.constrs == [c]
    assert x.model == m
    assert x.idx == 0

    assert x.rc is None
    assert x.x is None

    m.optimize()

    # Check after optimization
    assert abs(x.rc - 1.0) <= TOL
    assert abs(x.x - 5) < TOL
    assert float(x) == x.x

    # TODO check Xn in case of additional (sub-optimal) solutions


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_setting_variable_attributes(solver):
    m = Model(solver_name=solver, sense=MAXIMIZE)
    x = m.add_var("x")
    x.lb = -1.0
    x.ub = 5.5
    x.obj = 1.0
    x.var_type = INTEGER
    x.branch_priority = 1

    y = m.add_var("y", obj=1)
    c = m.add_constr(y <= x, "some_constraint")
    # TODO: Remove Not implemented error when implemented
    column = Column([c], [-2])  # new column based on constraint (y <= 2*x)
    if solver == HIGHS:
        x.column = column
    else:
        with pytest.raises(NotImplementedError):
            x.column = column

    # Check before optimization
    assert x.lb == -1.0
    assert x.ub == 5.5
    assert x.obj == 1.0
    assert x.var_type == INTEGER
    # As branch priority is currently not supported by CBC
    if solver == GUROBI:
        assert x.branch_priority == 1
    if solver == CBC:
        assert x.branch_priority == 0
    # TODO: Check when implemented
    if solver == HIGHS:
        column = x.column
        assert column.coeffs == [-2]
        assert column.constrs == [c]

    m.optimize()

    # Check that optimization result considered changes correctly
    if solver == HIGHS:
        # column was changed, so y == 2*x
        assert abs(m.objective_value - 15.0) <= TOL
        assert abs(x.x - 5) < TOL
        assert abs(y.x - 10) < TOL
    else:
        # column was not changed, so y == x
        assert abs(m.objective_value - 10.0) <= TOL
        assert abs(x.x - 5) < TOL
        assert abs(y.x - 5) < TOL


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_forbidden_overwrites_of_variable_attributes(solver):
    m = Model(solver_name=solver)
    x = m.add_var("x")

    overwrite_model = Model(solver_name=solver)
    with pytest.raises(AttributeError):
        x.model = overwrite_model

    with pytest.raises(AttributeError):
        x.idx = 1

    with pytest.raises(AttributeError):
        x.x = 5

    with pytest.raises(AttributeError):
        x.name = "y"

    with pytest.raises(AttributeError):
        x.rc = 6


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_wrong_overwrites_of_variable_attributes(solver):
    m = Model(solver_name=solver)
    x = m.add_var("x")

    with pytest.raises(TypeError):
        x.obj = "1"

    with pytest.raises(TypeError):
        x.lb = "0"

    with pytest.raises(TypeError):
        x.ub = "1"

    # TODO: Check when implemented
    # with pytest.raises(TypeError):
    #    x.branch_priority = "1"

    with pytest.raises(ValueError):
        x.var_type = "0"

    # TODO: Check when implemented
    # with pytest.raises(TypeError):
    #    x.column = ["1", "1"]


# LinExpr Tests

@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("constant", (-1, 0, 1.5, 2))
def test_addition_of_lin_expr_with_constant(solver, constant):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    term_right = term + constant
    assert type(term_right) == LinExpr
    assert term_right.const == constant
    assert term_right.expr == {x: 1, y: 1}

    term_left = constant + term
    assert type(term_left) == LinExpr
    assert term_left.const == constant
    assert term_left.expr == {x: 1, y: 1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_addition_of_lin_expr_with_var(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    z = m.add_var(name="z")
    term = x + y

    term_right = term + z
    assert type(term_right) == LinExpr
    assert term_right.const == 0
    assert term_right.expr == {x: 1, y: 1, z: 1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_addition_of_lin_expr_with_lin_expr(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    a = m.add_var(name="a")
    b = m.add_var(name="b")
    term = x + y

    term_to_add = a + b

    added_term = term + term_to_add
    assert type(added_term) == LinExpr
    assert added_term.const == 0
    assert added_term.expr == {x: 1, y: 1, a: 1, b: 1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_addition_of_lin_expr_with_illegal_type(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    with pytest.raises(TypeError):
        term + "1"

    with pytest.raises(TypeError):
        "1" + term


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("constant", (-1, 0, 1.5, 2))
def test_inplace_addition_of_lin_expr_with_constant(solver, constant):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    term += constant

    assert type(term) == LinExpr
    assert term.const == constant
    assert term.expr == {x: 1, y: 1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_inplace_addition_of_lin_expr_with_var(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    z = m.add_var(name="z")
    term = x + y

    term += z
    assert type(term) == LinExpr
    assert term.const == 0
    assert term.expr == {x: 1, y: 1, z: 1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_inplace_addition_of_lin_expr_with_lin_expr(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    a = m.add_var(name="a")
    b = m.add_var(name="b")
    term = x + y

    term_to_add = a + b

    term += term_to_add
    assert type(term) == LinExpr
    assert term.const == 0
    assert term.expr == {x: 1, y: 1, a: 1, b: 1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_inplace_addition_of_lin_expr_with_illegal_type(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    with pytest.raises(TypeError):
        term += "1"


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("constant", (-1, 0, 1.5, 2))
def test_subtraction_of_lin_expr_and_constant(solver, constant):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    term_right = term - constant
    assert type(term_right) == LinExpr
    assert term_right.const == -constant
    assert term_right.expr == {x: 1, y: 1}

    term_left = constant - term
    assert type(term_left) == LinExpr
    assert term_left.const == constant
    assert term_left.expr == {x: -1, y: -1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_subtraction_of_lin_expr_and_var(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    z = m.add_var(name="z")
    term = x + y

    term_right = term - z
    assert type(term_right) == LinExpr
    assert term_right.const == 0
    assert term_right.expr == {x: 1, y: 1, z: -1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_subtraction_of_lin_expr_with_lin_expr(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    a = m.add_var(name="a")
    b = m.add_var(name="b")
    term = x + y

    term_to_sub = a + b

    sub_term = term - term_to_sub
    assert type(sub_term) == LinExpr
    assert sub_term.const == 0
    assert sub_term.expr == {x: 1, y: 1, a: -1, b: -1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_subtraction_of_lin_expr_and_illegal_type(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    with pytest.raises(TypeError):
        term - "1"

    with pytest.raises(TypeError):
        "1" - term


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("constant", (-1, 0, 1.5, 2))
def test_inplace_subtraction_of_lin_expr_and_constant(solver, constant):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    term -= constant

    assert type(term) == LinExpr
    assert term.const == -constant
    assert term.expr == {x: 1, y: 1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_inplace_subtraction_of_lin_expr_and_var(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    z = m.add_var(name="z")
    term = x + y

    term -= z
    assert type(term) == LinExpr
    assert term.const == 0
    assert term.expr == {x: 1, y: 1, z: -1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_inplace_subtraction_of_lin_expr_and_lin_expr(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    a = m.add_var(name="a")
    b = m.add_var(name="b")
    term = x + y

    term_to_sub = a + b

    term -= term_to_sub
    assert type(term) == LinExpr
    assert term.const == 0
    assert term.expr == {x: 1, y: 1, a: -1, b: -1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_inplace_subtraction_of_lin_expr_and_illegal_type(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    with pytest.raises(TypeError):
        term -= "1"


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("coefficient", (-1, 0, 1.2, 2))
def test_multiply_lin_expr_with_coefficient(solver, coefficient):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    right = coefficient * term
    assert type(right) == LinExpr
    assert right.const == 0.0
    assert right.expr == {x: coefficient, y: coefficient}

    left = term * coefficient
    assert type(left) == LinExpr
    assert left.const == 0.0
    assert left.expr == {x: coefficient, y: coefficient}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_multiply_lin_expr_with_illegal_coefficient(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    with pytest.raises(TypeError):
        "1" * term

    with pytest.raises(TypeError):
        term * "1"


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("coefficient", (-1, 0, 1.2, 2))
def test_inplace_multiplication_lin_expr_with_coefficient(solver, coefficient):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    term *= coefficient
    assert type(term) == LinExpr
    assert term.const == 0.0
    assert term.expr == {x: coefficient, y: coefficient}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_inplace_multiplication_lin_expr_with_illegal_coefficient(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    with pytest.raises(TypeError):
        term *= "1"


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("coefficient", (-1, 1.2, 2))
def test_division_lin_expr_non_zero_coefficient(solver, coefficient):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    right = term / coefficient
    assert type(right) == LinExpr
    assert right.const == 0.0
    assert right.expr == {x: 1 / coefficient, y: 1 / coefficient}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_divide_lin_expr_with_illegal_coefficient(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    with pytest.raises(TypeError):
        term / "1"


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_divide_lin_expr_by_zero(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    with pytest.raises(ZeroDivisionError):
        term / 0


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("coefficient", (-1, 1.2, 2))
def test_inplace_division_lin_expr_non_zero_coefficient(solver, coefficient):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    term /= coefficient
    assert type(term) == LinExpr
    assert term.const == 0.0
    assert term.expr == {x: 1 / coefficient, y: 1 / coefficient}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_inplace_division_lin_expr_by_zero(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    with pytest.raises(ZeroDivisionError):
        term /= 0


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_inplace_division_lin_expr_with_illegal_coefficient(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    with pytest.raises(TypeError):
        term /= "1"


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("coefficient", (-1, 1.2, 2))
def test_negating_lin_expr_non_zero_coefficient(solver, coefficient):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + y

    neg = - term
    assert type(neg) == LinExpr
    assert neg.const == 0.0
    assert neg.expr == {x: -1, y: -1}


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_len_of_lin_expr(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")

    assert len(LinExpr()) == 0
    assert len(x + y) == 2
    assert len(x + y + 1) == 2


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("coefficient", (-2, 0, 1.1, 3))
def test_add_term_with_valid_input(solver, coefficient):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    a = m.add_var(name="a")
    b = m.add_var(name="b")

    # Test constant
    term = x + y
    add_term = 1
    term.add_term(add_term, coefficient)
    assert term.expr == {x: 1, y: 1}
    assert term.const == coefficient

    # Test variable
    term = x + y
    add_term = a
    term.add_term(add_term, coefficient)
    assert term.expr == {x: 1, y: 1, a: coefficient}
    assert term.const == 0.0

    # Test expression
    term = x + y
    add_term = a + b + 1
    term.add_term(add_term, coefficient)
    assert term.expr == {x: 1, y: 1, a: coefficient, b: coefficient}
    assert term.const == coefficient

    # Test illegal
    term = x + y
    add_term = "1"
    with pytest.raises(TypeError):
        term.add_term(add_term, coefficient)


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_hash(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    term = x + 2*y + 3

    assert type(hash(term)) == int


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_copy(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")

    term = x + y + 1
    term_copy = term.copy()

    assert term.const == term_copy.const
    assert term.sense == term_copy.sense
    assert term.expr == term_copy.expr
    assert id(term.expr) != id(term_copy.expr)


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_verbose(solver):
    # set and get verbose flag
    m = Model(solver_name=solver)

    # active
    m.verbose = 1
    assert m.verbose == 1

    # inactive
    m.verbose = 0
    assert m.verbose == 0


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_constraint_with_lin_expr_and_lin_expr(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    a = m.add_var(name="a")
    b = m.add_var(name="b")

    term = x + y
    other_term = a + b + 1

    constr = term <= other_term
    assert type(constr) == LinExpr
    assert constr.const == -1
    assert constr.expr == {x: 1, y: 1, a: -1, b: -1}
    assert constr.sense == "<"

    constr = term == other_term
    assert type(constr) == LinExpr
    assert constr.const == -1
    assert constr.expr == {x: 1, y: 1, a: -1, b: -1}
    assert constr.sense == "="

    constr = term >= other_term
    assert type(constr) == LinExpr
    assert constr.const == -1
    assert constr.expr == {x: 1, y: 1, a: -1, b: -1}
    assert constr.sense == ">"


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_query_attributes_of_lin_expr(solver):
    m = Model(solver_name=solver, sense=MAXIMIZE)
    x = m.add_var(name="x", ub=5)
    y = m.add_var(name="y", ub=2)

    term = x + y - 1

    # Before optimization and set as constraint
    assert term.sense == ""
    assert term.x is None
    assert term.model == m
    assert term.violation is None

    constr_expr = term <= 5
    m.add_constr(constr_expr, name="a_constraint")

    # Before optimization
    assert constr_expr.sense == "<"
    assert constr_expr.x is None
    assert constr_expr.violation is None

    m.optimize()

@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_objective(solver):
    m = Model(solver_name=solver, sense=MAXIMIZE)
    x = m.add_var(name="x", lb=0, ub=1)
    y = m.add_var(name="y", lb=0, ub=1)
    z = m.add_var(name="z", lb=0, ub=1)

    m.objective = x - y + 0.5
    assert m.objective.x is None
    #TODO: assert m.objective.sense == MAXIMIZE

    # Make sure that we can access the objective and it's correct
    assert len(m.objective.expr) == 2
    assert m.objective.expr[x] == 1
    assert m.objective.expr[y] == -1
    assert m.objective.const == 0.5

    status = m.optimize()
    assert status == OptimizationStatus.OPTIMAL
    assert m.objective_value == 1.5
    assert m.objective_value == m.objective.x


    # Test changing the objective
    m.objective = y + 2*z + 1.5
    m.sense = MINIMIZE
    # TODO: assert m.objective.sense == MINIMIZE

    assert len(m.objective.expr) == 2
    assert m.objective.expr[y] == 1
    assert m.objective.expr[z] == 2
    assert m.objective.const == 1.5

    status = m.optimize()
    assert status == OptimizationStatus.OPTIMAL
    assert m.objective_value == 1.5
    assert m.objective_value == m.objective.x

@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_remove(solver):
    m = Model(solver_name=solver)
    x = m.add_var("x")
    constr = m.add_constr(x >= 0)
    m.objective = x

    with pytest.raises(TypeError, match=re.escape("Cannot handle removal of object of type <class 'NoneType'> from model")):
        m.remove(None)

    with pytest.raises(TypeError, match=re.escape("Cannot handle removal of object of type <class 'NoneType'> from model")):
        m.remove([None])

    m.remove(constr)
    m.remove(x)

@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_add_equation(solver):
    m = Model(solver_name=solver)
    x = m.add_var("x")
    constr = m.add_constr(x == 23.5)

    status = m.optimize()
    assert status == OptimizationStatus.OPTIMAL

    assert x.x == pytest.approx(23.5)

@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_change_objective_sense(solver):
    m = Model(solver_name=solver)
    x = m.add_var("x", lb=10.0, ub=20.0)

    # first maximize
    m.objective = mip.maximize(x)
    status = m.optimize()
    assert status == OptimizationStatus.OPTIMAL
    assert x.x == pytest.approx(20.0)

    # then minimize
    m.objective = mip.minimize(x)
    status = m.optimize()
    assert status == OptimizationStatus.OPTIMAL
    assert x.x == pytest.approx(10.0)

@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_solve_relaxation(solver):
    m = Model(solver_name=solver)
    x = m.add_var("x", var_type=CONTINUOUS)
    y = m.add_var("y", var_type=INTEGER)
    z = m.add_var("z", var_type=BINARY)

    c1 = m.add_constr(x <= 10 * z)
    c2 = m.add_constr(x <= 9.5)
    c3 = m.add_constr(x + y <= 20)
    m.objective = mip.maximize(4*x + y - z)

    # double-check constraint expressions
    assert c1.idx == 0
    expr1 = c1.expr  # store to avoid repeated calls
    assert expr1.expr == pytest.approx({x: 1.0, z: -10.0})
    assert expr1.const == pytest.approx(0.0)
    assert expr1.sense == mip.LESS_OR_EQUAL

    # first solve proper MIP
    status = m.optimize()
    assert status == OptimizationStatus.OPTIMAL
    assert x.x == pytest.approx(9.5)
    assert y.x == pytest.approx(10.0)
    assert z.x == pytest.approx(1.0)

    assert c1.slack == pytest.approx(0.5)
    assert c2.slack == pytest.approx(0.0)
    assert c3.slack == pytest.approx(0.5)

    # then compare LP relaxation
    # (seems to fail for CBC?!)
    if solver == HIGHS:
        status = m.optimize(relax=True)
        assert status == OptimizationStatus.OPTIMAL
        assert x.x == pytest.approx(9.5)
        assert y.x == pytest.approx(10.5)
        assert z.x == pytest.approx(0.95)

        assert c1.slack == pytest.approx(0.0)
        assert c2.slack == pytest.approx(0.0)
        assert c3.slack == pytest.approx(0.0)
