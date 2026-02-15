"""Tests for Python-MIP"""
import math
from itertools import product
from os import environ

import networkx as nx
import mip.gurobi
import mip.highs
from mip import Model, xsum, OptimizationStatus, MAXIMIZE, BINARY, INTEGER
from mip import ConstrsGenerator, CutPool, maximize, CBC, GUROBI, HIGHS, Column, Constr
from os import environ
from util import skip_on
import math
import pytest

TOL = 1e-4

SOLVERS = [CBC]
if mip.gurobi.has_gurobi and "GUROBI_HOME" in environ:
    SOLVERS += [GUROBI]
if mip.highs.has_highs:
    SOLVERS += [HIGHS]


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_column_generation(solver: str):
    L = 250  # bar length
    m = 4  # number of requests
    w = [187, 119, 74, 90]  # size of each item
    b = [1, 2, 2, 1]  # demand for each item

    # creating master model
    master = Model(solver_name=solver)

    # creating an initial set of patterns which cut one item per bar
    # to provide the restricted master problem with a feasible solution
    lambdas = [master.add_var(obj=1, name="lambda_%d" % (j + 1)) for j in range(m)]

    # creating constraints
    constraints = []
    for i in range(m):
        constraints.append(master.add_constr(lambdas[i] >= b[i], name="i_%d" % (i + 1)))

    # creating the pricing problem
    pricing = Model(solver_name=solver)

    # creating pricing variables
    a = [
        pricing.add_var(obj=0, var_type=INTEGER, name="a_%d" % (i + 1)) for i in range(m)
    ]

    # creating pricing constraint
    pricing += xsum(w[i] * a[i] for i in range(m)) <= L, "bar_length"

    new_vars = True
    while new_vars:
        ##########
        # STEP 1: solving restricted master problem
        ##########
        master.optimize()

        ##########
        # STEP 2: updating pricing objective with dual values from master
        ##########
        pricing += 1 - xsum(constraints[i].pi * a[i] for i in range(m))

        # solving pricing problem
        pricing.optimize()

        ##########
        # STEP 3: adding the new columns (if any is obtained with negative reduced cost)
        ##########
        # checking if columns with negative reduced cost were produced and
        # adding them into the restricted master problem
        if pricing.objective_value < -TOL:
            pattern = [a[i].x for i in range(m)]
            column = Column(constraints, pattern)
            lambdas.append(
                master.add_var(
                    obj=1, column=column, name="lambda_%d" % (len(lambdas) + 1)
                )
            )

        # if no column with negative reduced cost was produced, then linear
        # relaxation of the restricted master problem is solved
        else:
            new_vars = False

    # printing the solution
    assert len(lambdas) == 8
    assert round(master.objective_value) == 3


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_cutting_stock(solver: str):
    n = 10  # maximum number of bars
    L = 250  # bar length
    m = 4  # number of requests
    w = [187, 119, 74, 90]  # size of each item
    b = [1, 2, 2, 1]  # demand for each item

    # creating the model
    model = Model(solver_name=solver)
    x = {
        (i, j): model.add_var(obj=0, var_type=INTEGER, name="x[%d,%d]" % (i, j))
        for i in range(m)
        for j in range(n)
    }
    y = {j: model.add_var(obj=1, var_type=BINARY, name="y[%d]" % j) for j in range(n)}

    # constraints
    for i in range(m):
        model.add_constr(xsum(x[i, j] for j in range(n)) >= b[i])
    for j in range(n):
        model.add_constr(xsum(w[i] * x[i, j] for i in range(m)) <= L * y[j])

    # additional constraints to reduce symmetry
    for j in range(1, n):
        model.add_constr(y[j - 1] >= y[j])

    # optimizing the model
    model.optimize()

    # sanity tests
    assert model.status == OptimizationStatus.OPTIMAL
    assert abs(model.objective_value - 3) <= 1e-4
    assert sum(x.x for x in model.vars) >= 5


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_knapsack(solver: str):
    p = [10, 13, 18, 31, 7, 15]
    w = [11, 15, 20, 35, 10, 33]
    c, I = 47, range(len(w))

    m = Model("knapsack", solver_name=solver)

    x = [m.add_var(var_type=BINARY) for i in I]

    m.objective = maximize(xsum(p[i] * x[i] for i in I))

    m += xsum(w[i] * x[i] for i in I) <= c, "cap"

    m.optimize()

    assert m.status == OptimizationStatus.OPTIMAL
    assert round(m.objective_value) == 41

    m.constr_by_name("cap").rhs = 60
    m.optimize()

    assert m.status == OptimizationStatus.OPTIMAL
    assert round(m.objective_value) == 51

    # modifying objective function
    m.objective = m.objective + 10 * x[0] + 15 * x[1]
    assert abs(m.objective.expr[x[0]] - 20) <= 1e-10
    assert abs(m.objective.expr[x[1]] - 28) <= 1e-10


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_queens(solver: str):
    """MIP model n-queens"""
    n = 50
    queens = Model("queens", MAXIMIZE, solver_name=solver)
    queens.verbose = 0

    x = [
        [queens.add_var("x({},{})".format(i, j), var_type=BINARY) for j in range(n)]
        for i in range(n)
    ]

    # one per row
    for i in range(n):
        queens += xsum(x[i][j] for j in range(n)) == 1, "row({})".format(i)

    # one per column
    for j in range(n):
        queens += xsum(x[i][j] for i in range(n)) == 1, "col({})".format(j)

    # diagonal \
    for p, k in enumerate(range(2 - n, n - 2 + 1)):
        queens += (
            xsum(x[i][j] for i in range(n) for j in range(n) if i - j == k) <= 1,
            "diag1({})".format(p),
        )

    # diagonal /
    for p, k in enumerate(range(3, n + n)):
        queens += (
            xsum(x[i][j] for i in range(n) for j in range(n) if i + j == k) <= 1,
            "diag2({})".format(p),
        )

    queens.optimize()
    assert queens.status == OptimizationStatus.OPTIMAL  # "model status"

    # querying problem variables and checking opt results
    total_queens = 0
    for v in queens.vars:
        # basic integrality test
        assert v.x <= TOL or v.x >= 1 - TOL
        total_queens += v.x

    # solution feasibility
    rows_with_queens = 0
    for i in range(n):
        if abs(sum(x[i][j].x for j in range(n)) - 1) <= TOL:
            rows_with_queens += 1

    assert abs(total_queens - n) <= TOL  # "feasible solution"
    assert rows_with_queens == n  # "feasible solution"


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_tsp(solver: str):
    """tsp related tests"""
    N = ["a", "b", "c", "d", "e", "f", "g"]
    n = len(N)
    i0 = N[0]

    A = {
        ("a", "d"): 56,
        ("d", "a"): 67,
        ("a", "b"): 49,
        ("b", "a"): 50,
        ("d", "b"): 39,
        ("b", "d"): 37,
        ("c", "f"): 35,
        ("f", "c"): 35,
        ("g", "b"): 35,
        ("b", "g"): 25,
        ("a", "c"): 80,
        ("c", "a"): 99,
        ("e", "f"): 20,
        ("f", "e"): 20,
        ("g", "e"): 38,
        ("e", "g"): 49,
        ("g", "f"): 37,
        ("f", "g"): 32,
        ("b", "e"): 21,
        ("e", "b"): 30,
        ("a", "g"): 47,
        ("g", "a"): 68,
        ("d", "c"): 37,
        ("c", "d"): 52,
        ("d", "e"): 15,
        ("e", "d"): 20,
    }

    # input and output arcs per node
    Aout = {n: [a for a in A if a[0] == n] for n in N}
    Ain = {n: [a for a in A if a[1] == n] for n in N}
    m = Model(solver_name=solver)
    m.verbose = 1

    x = {a: m.add_var(name="x({},{})".format(a[0], a[1]), var_type=BINARY) for a in A}

    m.objective = xsum(c * x[a] for a, c in A.items())

    for i in N:
        m += xsum(x[a] for a in Aout[i]) == 1, "out({})".format(i)
        m += xsum(x[a] for a in Ain[i]) == 1, "in({})".format(i)

    # continuous variable to prevent subtours: each
    # city will have a different "identifier" in the planned route
    y = {i: m.add_var(name="y({})".format(i), lb=0.0) for i in N}

    # subtour elimination
    for (i, j) in A:
        if i0 not in [i, j]:
            m.add_constr(y[i] - (n + 1) * x[(i, j)] >= y[j] - n)

    m.relax()
    m.optimize()

    assert m.status == OptimizationStatus.OPTIMAL  # "lp model status"
    assert abs(m.objective_value - 238.75) <= TOL  # "lp model objective"

    # setting all variables to integer now
    for v in m.vars:
        v.var_type = INTEGER
    m.optimize()

    assert m.status == OptimizationStatus.OPTIMAL  # "mip model status"
    assert abs(m.objective_value - 262) <= TOL  # "mip model objective"


class SubTourCutGenerator(ConstrsGenerator):
    """Class to generate cutting planes for the TSP"""

    def generate_constrs(self, model: Model, depth: int = 0, npass: int = 0):
        G = nx.DiGraph()
        r = [(v, v.x) for v in model.vars if v.name.startswith("x(")]
        U = [v.name.split("(")[1].split(",")[0] for v, f in r]
        V = [v.name.split(")")[0].split(",")[1] for v, f in r]
        N = list(set(U + V))
        cp = CutPool()
        for i in range(len(U)):
            G.add_edge(U[i], V[i], capacity=r[i][1])
        for (u, v) in product(N, N):
            if u == v:
                continue
            val, (S, NS) = nx.minimum_cut(G, u, v)
            if val <= 0.99:
                arcsInS = [
                    (v, f) for i, (v, f) in enumerate(r) if U[i] in S and V[i] in S
                ]
                if sum(f for v, f in arcsInS) >= (len(S) - 1) + 1e-4:
                    cut = xsum(1.0 * v for v, fm in arcsInS) <= len(S) - 1
                    cp.add(cut)
                    if len(cp.cuts) > 256:
                        for cut in cp.cuts:
                            model.add_cut(cut)
                        return
        for cut in cp.cuts:
            model.add_cut(cut)


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_tsp_cuts(solver: str):
    """tsp related tests"""
    N = ["a", "b", "c", "d", "e", "f", "g"]
    n = len(N)
    i0 = N[0]

    A = {
        ("a", "d"): 56,
        ("d", "a"): 67,
        ("a", "b"): 49,
        ("b", "a"): 50,
        ("d", "b"): 39,
        ("b", "d"): 37,
        ("c", "f"): 35,
        ("f", "c"): 35,
        ("g", "b"): 35,
        ("b", "g"): 25,
        ("a", "c"): 80,
        ("c", "a"): 99,
        ("e", "f"): 20,
        ("f", "e"): 20,
        ("g", "e"): 38,
        ("e", "g"): 49,
        ("g", "f"): 37,
        ("f", "g"): 32,
        ("b", "e"): 21,
        ("e", "b"): 30,
        ("a", "g"): 47,
        ("g", "a"): 68,
        ("d", "c"): 37,
        ("c", "d"): 52,
        ("d", "e"): 15,
        ("e", "d"): 20,
    }

    # input and output arcs per node
    Aout = {n: [a for a in A if a[0] == n] for n in N}
    Ain = {n: [a for a in A if a[1] == n] for n in N}
    m = Model(solver_name=solver)
    m.verbose = 0

    x = {a: m.add_var(name="x({},{})".format(a[0], a[1]), var_type=BINARY) for a in A}

    m.objective = xsum(c * x[a] for a, c in A.items())

    for i in N:
        m += xsum(x[a] for a in Aout[i]) == 1, "out({})".format(i)
        m += xsum(x[a] for a in Ain[i]) == 1, "in({})".format(i)

    # continuous variable to prevent subtours: each
    # city will have a different "identifier" in the planned route
    y = {i: m.add_var(name="y({})".format(i), lb=0.0) for i in N}

    # subtour elimination
    for (i, j) in A:
        if i0 not in [i, j]:
            m.add_constr(y[i] - (n + 1) * x[(i, j)] >= y[j] - n)

    m.cuts_generator = SubTourCutGenerator()

    # tiny model, should be enough to find the optimal
    m.max_seconds = 10
    m.max_nodes = 100
    m.max_solutions = 1000
    m.optimize()

    assert m.status == OptimizationStatus.OPTIMAL  # "mip model status"
    assert abs(m.objective_value - 262) <= TOL  # "mip model objective"


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_tsp_mipstart(solver: str):
    """tsp related tests"""
    N = ["a", "b", "c", "d", "e", "f", "g"]
    n = len(N)
    i0 = N[0]

    A = {
        ("a", "d"): 56,
        ("d", "a"): 67,
        ("a", "b"): 49,
        ("b", "a"): 50,
        ("d", "b"): 39,
        ("b", "d"): 37,
        ("c", "f"): 35,
        ("f", "c"): 35,
        ("g", "b"): 35,
        ("b", "g"): 25,
        ("a", "c"): 80,
        ("c", "a"): 99,
        ("e", "f"): 20,
        ("f", "e"): 20,
        ("g", "e"): 38,
        ("e", "g"): 49,
        ("g", "f"): 37,
        ("f", "g"): 32,
        ("b", "e"): 21,
        ("e", "b"): 30,
        ("a", "g"): 47,
        ("g", "a"): 68,
        ("d", "c"): 37,
        ("c", "d"): 52,
        ("d", "e"): 15,
        ("e", "d"): 20,
    }

    # input and output arcs per node
    Aout = {n: [a for a in A if a[0] == n] for n in N}
    Ain = {n: [a for a in A if a[1] == n] for n in N}
    m = Model(solver_name=solver)
    m.verbose = 0

    x = {a: m.add_var(name="x({},{})".format(a[0], a[1]), var_type=BINARY) for a in A}

    m.objective = xsum(c * x[a] for a, c in A.items())

    for i in N:
        m += xsum(x[a] for a in Aout[i]) == 1, "out({})".format(i)
        m += xsum(x[a] for a in Ain[i]) == 1, "in({})".format(i)

    # continuous variable to prevent subtours: each
    # city will have a different "identifier" in the planned route
    y = {i: m.add_var(name="y({})".format(i), lb=0.0) for i in N}

    # subtour elimination
    for (i, j) in A:
        if i0 not in [i, j]:
            m.add_constr(y[i] - (n + 1) * x[(i, j)] >= y[j] - n)

    route = ["a", "g", "f", "c", "d", "e", "b", "a"]
    m.start = [(x[route[i - 1], route[i]], 1.0) for i in range(1, len(route))]
    m.optimize()

    assert m.status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - 262) <= TOL


class TestAPI(object):
    def build_model(self, solver):
        """MIP model n-queens"""
        n = 50
        queens = Model("queens", MAXIMIZE, solver_name=solver)
        queens.verbose = 0

        x = [
            [queens.add_var("x({},{})".format(i, j), var_type=BINARY) for j in range(n)]
            for i in range(n)
        ]

        # one per row
        for i in range(n):
            queens += xsum(x[i][j] for j in range(n)) == 1, "row{}".format(i)

        # one per column
        for j in range(n):
            queens += xsum(x[i][j] for i in range(n)) == 1, "col{}".format(j)

        # diagonal \
        for p, k in enumerate(range(2 - n, n - 2 + 1)):
            queens += (
                xsum(x[i][j] for i in range(n) for j in range(n) if i - j == k) <= 1,
                "diag1({})".format(p),
            )

        # diagonal /
        for p, k in enumerate(range(3, n + n)):
            queens += (
                xsum(x[i][j] for i in range(n) for j in range(n) if i + j == k) <= 1,
                "diag2({})".format(p),
            )

        return n, queens

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_constr_get_index(self, solver):
        _, model = self.build_model(solver)

        idx = model.solver.constr_get_index("row0")
        assert idx >= 0

        idx = model.solver.constr_get_index("col0")
        assert idx >= 0

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_remove_constrs(self, solver):
        _, model = self.build_model(solver)

        idx1 = model.solver.constr_get_index("row0")
        assert idx1 >= 0

        idx2 = model.solver.constr_get_index("col0")
        assert idx2 >= 0

        model.solver.remove_constrs([idx1, idx2])

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_constr_get_rhs(self, solver):
        n, model = self.build_model(solver)

        # test RHS of rows
        for i in range(n):
            idx1 = model.solver.constr_get_index("row{}".format(i))
            assert idx1 >= 0
            assert model.solver.constr_get_rhs(idx1) == 1

        # test RHS of columns
        for i in range(n):
            idx1 = model.solver.constr_get_index("col{}".format(i))
            assert idx1 >= 0
            assert model.solver.constr_get_rhs(idx1) == 1

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_constr_set_rhs(self, solver):
        n, model = self.build_model(solver)

        idx1 = model.solver.constr_get_index("row0")
        assert idx1 >= 0

        val = 10
        model.solver.constr_set_rhs(idx1, val)
        assert model.solver.constr_get_rhs(idx1) == val

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_constr_by_name_rhs(self, solver):
        n, model = self.build_model(solver)

        val = 10
        model.constr_by_name("row0").rhs = val
        assert model.constr_by_name("row0").rhs == val

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_var_by_name_valid(self, solver):
        n, model = self.build_model(solver)

        name = "x({},{})".format(0, 0)
        v = model.var_by_name(name)
        assert v is not None
        assert isinstance(v, mip.Var)
        assert v.name == name

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_var_by_name_invalid(self, solver):
        n, model = self.build_model(solver)

        v = model.var_by_name("xyz_invalid_name")
        assert v is None

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_obj_const1(self, solver: str):
        n, model = self.build_model(solver)

        model.objective = 1
        e = model.objective
        assert e.const == 1

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_obj_const2(self, solver: str):
        n, model = self.build_model(solver)

        model.objective = 1
        assert model.objective_const == 1


@skip_on(NotImplementedError)
@pytest.mark.parametrize("val", range(1, 4))
@pytest.mark.parametrize("solver", SOLVERS)
def test_variable_bounds(solver: str, val: int):
    m = Model("bounds", solver_name=solver)

    x = m.add_var(var_type=INTEGER, lb=0, ub=2 * val)
    y = m.add_var(var_type=INTEGER, lb=val, ub=2 * val)
    m.objective = maximize(x - y)
    m.optimize()
    assert m.status == OptimizationStatus.OPTIMAL
    assert round(m.objective_value) == val
    assert round(x.x) == 2 * val
    assert round(y.x) == val


@skip_on(NotImplementedError)
@pytest.mark.parametrize("val", range(1, 4))
@pytest.mark.parametrize("solver", SOLVERS)
def test_linexpr_x(solver: str, val: int):
    m = Model("bounds", solver_name=solver)

    x = m.add_var(lb=0, ub=2 * val)
    y = m.add_var(lb=val, ub=2 * val)
    obj = x - y

    assert obj.x is None  # No solution yet.

    m.objective = maximize(obj)
    m.optimize()

    assert m.status == OptimizationStatus.OPTIMAL
    assert round(m.objective_value) == val
    assert round(x.x) == 2 * val
    assert round(y.x) == val

    # Check that the linear expression value is equal to the same expression
    # calculated from the values of the variables.
    assert abs((x + y).x - (x.x + y.x)) < TOL
    assert abs((x + 2 * y).x - (x.x + 2 * y.x)) < TOL
    assert abs((x + 2 * y + x).x - (x.x + 2 * y.x + x.x)) < TOL
    assert abs((x + 2 * y + x + 1).x - (x.x + 2 * y.x + x.x + 1)) < TOL
    assert abs((x + 2 * y + x + 1 + x / 2).x - (x.x + 2 * y.x + x.x + 1 + x.x / 2)) < TOL


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_add_column(solver: str):
    """Simple test which add columns in a specific way"""
    m = Model(solver_name=solver)
    x = m.add_var()

    example_constr1 = m.add_constr(x >= 1, "constr1")
    example_constr2 = m.add_constr(x <= 2, "constr2")
    column1 = Column()
    column1.constrs = [example_constr1]
    column1.coeffs = [1]
    second_var = m.add_var("second", column=column1)
    column2 = Column()
    column2.constrs = [example_constr2]
    column2.coeffs = [2]
    m.add_var("third", column=column2)

    vthird = m.vars["third"]
    assert vthird is not None
    assert len(vthird.column.coeffs) == len(vthird.column.constrs)
    assert len(vthird.column.coeffs) == 1

    pconstr2 = m.constrs["constr2"]
    assert vthird.column.constrs[0].name == pconstr2.name
    assert len(example_constr1.expr.expr) == 2
    assert second_var in example_constr1.expr.expr
    assert x in example_constr1.expr.expr


@skip_on(NotImplementedError)
@pytest.mark.parametrize("val", range(1, 4))
@pytest.mark.parametrize("solver", SOLVERS)
def test_float(solver: str, val: int):
    m = Model("bounds", solver_name=solver)
    x = m.add_var(lb=0, ub=2 * val)
    y = m.add_var(lb=val, ub=2 * val)
    obj = x - y
    # No solution yet. __float__ MUST return a float type, so it returns nan.
    assert obj.x is None
    assert math.isnan(float(obj))
    m.objective = maximize(obj)
    m.optimize()
    assert m.status == OptimizationStatus.OPTIMAL
    # test vars.
    assert x.x == float(x)
    assert y.x == float(y)
    # test linear expressions.
    assert float(x + y) == (x + y).x


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_empty_useless_constraint_is_considered(solver: str):
    m = Model("empty_constraint", solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    m.add_constr(xsum([]) <= 1, name="c_empty")  # useless, empty constraint
    m.add_constr(x + y <= 5, name="c1")
    m.add_constr(2 * x + y <= 6, name="c2")
    m.objective = maximize(x + 2 * y)
    m.optimize()
    # check objective
    assert m.status == OptimizationStatus.OPTIMAL
    assert abs(m.objective.x - 10) < TOL
    # check that all names of constraints could be queried
    assert {c.name for c in m.constrs} == {"c1", "c2", "c_empty"}
    assert all(isinstance(m.constr_by_name(c_name), Constr) for c_name in ("c1", "c2", "c_empty"))


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver", SOLVERS)
def test_empty_contradictory_constraint_is_considered(solver: str):
    m = Model("empty_constraint", solver_name=solver)
    x = m.add_var(name="x")
    y = m.add_var(name="y")
    m.add_constr(xsum([]) <= -1, name="c_contra")  # contradictory empty constraint
    m.add_constr(x + y <= 5, name="c1")
    m.objective = maximize(x + 2 * y)
    m.optimize()
    # assert infeasibility of problem
    assert m.status in (OptimizationStatus.INF_OR_UNBD, OptimizationStatus.INFEASIBLE)
    # check that all names of constraints could be queried
    assert {c.name for c in m.constrs} == {"c1", "c_contra"}
    assert all(isinstance(m.constr_by_name(c_name), Constr) for c_name in ("c1", "c_contra"))
