import os

import pytest

from mip import (
    CBC,
    Column,
    GUROBI,
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

TOL = 1e-4
SOLVERS = [CBC]
if "GUROBI_HOME" in os.environ:
    SOLVERS += [GUROBI]

# Overall Optimization Tests


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
    assert abs(m.objective_value - (-1)) < TOL

    # Maximum Case
    m = Model(solver_name=solver, sense=MAXIMIZE)
    m.add_var(name="x", var_type=INTEGER, lb=-1.5, ub=3.5, obj=1)
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - 3) < TOL


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("sense", (MINIMIZE, MAXIMIZE))
def test_single_default_variable_with_nothing_to_do(solver, sense):
    m = Model(solver_name=solver, sense=sense)
    m.add_var(name="x")
    m.optimize()
    # check result
    assert m.status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value) < TOL


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

@pytest.mark.parametrize("solver", SOLVERS)
def test_hashes_of_variables(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")

    assert hash(x) == 0
    assert hash(y) == 1


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


@pytest.mark.parametrize("solver", SOLVERS)
def test_addition_of_var_with_illegal_type(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    with pytest.raises(TypeError):
        x + "1"

    with pytest.raises(TypeError):
        "1" + x


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


@pytest.mark.parametrize("solver", SOLVERS)
def test_subtraction_of_var_with_illegal_type(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    with pytest.raises(TypeError):
        x - "1"

    with pytest.raises(TypeError):
        "1" - x


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


@pytest.mark.parametrize("solver", SOLVERS)
def test_multiply_var_with_illegal_coefficient(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")
    with pytest.raises(TypeError):
        "1" * x

    with pytest.raises(TypeError):
        x * "1"


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


@pytest.mark.parametrize("solver", SOLVERS)
def test_divide_var_with_illegal_coefficient(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    with pytest.raises(TypeError):
        y = x / "1"


@pytest.mark.parametrize("solver", SOLVERS)
def test_divide_var_with_zero(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    with pytest.raises(ZeroDivisionError):
        x / 0


@pytest.mark.parametrize("solver", SOLVERS)
def test_negate_variable(solver):
    m = Model(solver_name=solver)
    x = m.add_var(name="x")

    y = - x
    assert isinstance(y, LinExpr)
    assert y.const == 0.0
    assert y.expr == {x: -1}


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
    with pytest.raises(NotImplementedError):
        x.column = Column([c], [-2])  # new column based on constraint (y <= 2*x)

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
    # column = x.column
    # assert column.coeffs == [-2]
    # assert column.constrs == [c]

    m.optimize()

    # Check that optimization result considered changes correctly
    assert abs(m.objective_value - 10.0) <= TOL
    assert abs(x.x - 5) < TOL


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
