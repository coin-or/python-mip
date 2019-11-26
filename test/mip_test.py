"""Tests for Python-MIP"""
from itertools import product
import pytest
import networkx as nx
from mip.model import Model, xsum
from mip.constants import OptimizationStatus, MAXIMIZE, BINARY, INTEGER
from mip.callbacks import ConstrsGenerator, CutPool


TOL = 1E-4
SOLVERS = ['CBC', 'Gurobi']


@pytest.mark.parametrize("solver", SOLVERS)
def test_queens(solver: str):
    """MIP model n-queens"""
    n = 50
    queens = Model('queens', MAXIMIZE, solver_name=solver)
    queens.verbose = 0

    x = [[queens.add_var('x({},{})'.format(i, j), var_type=BINARY)
          for j in range(n)] for i in range(n)]

    # one per row
    for i in range(n):
        queens += xsum(x[i][j] for j in range(n)) == 1, 'row({})'.format(i)

    # one per column
    for j in range(n):
        queens += xsum(x[i][j] for i in range(n)) == 1, 'col({})'.format(j)

    # diagonal \
    for p, k in enumerate(range(2 - n, n - 2 + 1)):
        queens += xsum(x[i][j] for i in range(n) for j in range(n)
                       if i - j == k) <= 1, 'diag1({})'.format(p)

    # diagonal /
    for p, k in enumerate(range(3, n + n)):
        queens += xsum(x[i][j] for i in range(n) for j in range(n)
                       if i + j == k) <= 1, 'diag2({})'.format(p)

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
    assert rows_with_queens == n         # "feasible solution"


@pytest.mark.parametrize("solver", SOLVERS)
def test_tsp(solver: str):
    """tsp related tests"""
    N = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    n = len(N)
    i0 = N[0]

    A = {('a', 'd'): 56, ('d', 'a'): 67, ('a', 'b'): 49, ('b', 'a'): 50,
         ('d', 'b'): 39, ('b', 'd'): 37, ('c', 'f'): 35, ('f', 'c'): 35,
         ('g', 'b'): 35, ('b', 'g'): 25,
         ('a', 'c'): 80, ('c', 'a'): 99, ('e', 'f'): 20, ('f', 'e'): 20,
         ('g', 'e'): 38, ('e', 'g'): 49, ('g', 'f'): 37, ('f', 'g'): 32,
         ('b', 'e'): 21, ('e', 'b'): 30, ('a', 'g'): 47, ('g', 'a'): 68,
         ('d', 'c'): 37, ('c', 'd'): 52, ('d', 'e'): 15, ('e', 'd'): 20}

    # input and output arcs per node
    Aout = {n: [a for a in A if a[0] == n] for n in N}
    Ain = {n: [a for a in A if a[1] == n] for n in N}
    m = Model(solver_name=solver)
    m.verbose = 1

    x = {a: m.add_var(name='x({},{})'.format(a[0], a[1]),
                      var_type=BINARY) for a in A}

    m.objective = xsum(c * x[a] for a, c in A.items())

    for i in N:
        m += xsum(x[a] for a in Aout[i]) == 1, 'out({})'.format(i)
        m += xsum(x[a] for a in Ain[i]) == 1, 'in({})'.format(i)

    # continuous variable to prevent subtours: each
    # city will have a different "identifier" in the planned route
    y = {i: m.add_var(name='y({})'.format(i), lb=0.0) for i in N}

    # subtour elimination
    for (i, j) in A:
        if i0 not in [i, j]:
            m.add_constr(
                y[i] - (n + 1) * x[(i, j)] >= y[j] - n)

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

    def generate_constrs(self, model: Model):
        G = nx.DiGraph()
        r = [(v, v.x) for v in model.vars if v.name.startswith('x(')]
        U = [v.name.split('(')[1].split(',')[0] for v, f in r]
        V = [v.name.split(')')[0].split(',')[1] for v, f in r]
        N = list(set(U + V))
        cp = CutPool()
        for i in range(len(U)):
            G.add_edge(U[i], V[i], capacity=r[i][1])
        for (u, v) in product(N, N):
            if u == v:
                continue
            val, (S, NS) = nx.minimum_cut(G, u, v)
            if val <= 0.99:
                arcsInS = [(v, f) for i, (v, f) in enumerate(r)
                           if U[i] in S and V[i] in S]
                if sum(f for v, f in arcsInS) >= (len(S) - 1) + 1e-4:
                    cut = xsum(1.0 * v for v, fm in arcsInS) <= len(S) - 1
                    cp.add(cut)
                    if len(cp.cuts) > 256:
                        for cut in cp.cuts:
                            model.add_cut(cut)
                        return
        for cut in cp.cuts:
            model.add_cut(cut)


@pytest.mark.parametrize("solver", SOLVERS)
def test_tsp_cuts(solver: str):
    """tsp related tests"""
    N = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    n = len(N)
    i0 = N[0]

    A = {('a', 'd'): 56, ('d', 'a'): 67, ('a', 'b'): 49, ('b', 'a'): 50,
         ('d', 'b'): 39, ('b', 'd'): 37, ('c', 'f'): 35, ('f', 'c'): 35,
         ('g', 'b'): 35, ('b', 'g'): 25,
         ('a', 'c'): 80, ('c', 'a'): 99, ('e', 'f'): 20, ('f', 'e'): 20,
         ('g', 'e'): 38, ('e', 'g'): 49, ('g', 'f'): 37, ('f', 'g'): 32,
         ('b', 'e'): 21, ('e', 'b'): 30, ('a', 'g'): 47, ('g', 'a'): 68,
         ('d', 'c'): 37, ('c', 'd'): 52, ('d', 'e'): 15, ('e', 'd'): 20}

    # input and output arcs per node
    Aout = {n: [a for a in A if a[0] == n] for n in N}
    Ain = {n: [a for a in A if a[1] == n] for n in N}
    m = Model(solver_name=solver)
    m.verbose = 0

    x = {a: m.add_var(name='x({},{})'.format(a[0], a[1]),
                      var_type=BINARY) for a in A}

    m.objective = xsum(c * x[a] for a, c in A.items())

    for i in N:
        m += xsum(x[a] for a in Aout[i]) == 1, 'out({})'.format(i)
        m += xsum(x[a] for a in Ain[i]) == 1, 'in({})'.format(i)

    # continuous variable to prevent subtours: each
    # city will have a different "identifier" in the planned route
    y = {i: m.add_var(name='y({})'.format(i), lb=0.0) for i in N}

    # subtour elimination
    for (i, j) in A:
        if i0 not in [i, j]:
            m.add_constr(
                y[i] - (n + 1) * x[(i, j)] >= y[j] - n)

    m.cuts_generator = SubTourCutGenerator()

    # tiny model, should be enough to find the optimal
    m.max_seconds = 10
    m.max_nodes = 100
    m.max_solutions = 1000
    m.optimize()

    assert m.status == OptimizationStatus.OPTIMAL  # "mip model status"
    assert abs(m.objective_value - 262) <= TOL     # "mip model objective"


@pytest.mark.parametrize("solver", SOLVERS)
def test_tsp_mipstart(solver: str):
    """tsp related tests"""
    N = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    n = len(N)
    i0 = N[0]

    A = {('a', 'd'): 56, ('d', 'a'): 67, ('a', 'b'): 49, ('b', 'a'): 50,
         ('d', 'b'): 39, ('b', 'd'): 37, ('c', 'f'): 35, ('f', 'c'): 35,
         ('g', 'b'): 35, ('b', 'g'): 25,
         ('a', 'c'): 80, ('c', 'a'): 99, ('e', 'f'): 20, ('f', 'e'): 20,
         ('g', 'e'): 38, ('e', 'g'): 49, ('g', 'f'): 37, ('f', 'g'): 32,
         ('b', 'e'): 21, ('e', 'b'): 30, ('a', 'g'): 47, ('g', 'a'): 68,
         ('d', 'c'): 37, ('c', 'd'): 52, ('d', 'e'): 15, ('e', 'd'): 20}

    # input and output arcs per node
    Aout = {n: [a for a in A if a[0] == n] for n in N}
    Ain = {n: [a for a in A if a[1] == n] for n in N}
    m = Model(solver_name=solver)
    m.verbose = 0

    x = {a: m.add_var(name='x({},{})'.format(a[0], a[1]),
                      var_type=BINARY) for a in A}

    m.objective = xsum(c * x[a] for a, c in A.items())

    for i in N:
        m += xsum(x[a] for a in Aout[i]) == 1, 'out({})'.format(i)
        m += xsum(x[a] for a in Ain[i]) == 1, 'in({})'.format(i)

    # continuous variable to prevent subtours: each
    # city will have a different "identifier" in the planned route
    y = {i: m.add_var(name='y({})'.format(i), lb=0.0) for i in N}

    # subtour elimination
    for (i, j) in A:
        if i0 not in [i, j]:
            m.add_constr(
                y[i] - (n + 1) * x[(i, j)] >= y[j] - n)

    route = ['a', 'g', 'f', 'c', 'd', 'e', 'b', 'a']
    m.start = [(x[route[i - 1], route[i]], 1.0) for i in range(1, len(route))]
    m.optimize()

    assert m.status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - 262) <= TOL


class TestCBC(object):

    def build_model(self, solver):
        """MIP model n-queens"""
        n = 50
        queens = Model('queens', MAXIMIZE, solver_name=solver)
        queens.verbose = 0

        x = [[queens.add_var('x({},{})'.format(i, j), var_type=BINARY)
              for j in range(n)] for i in range(n)]

        # one per row
        for i in range(n):
            queens += xsum(x[i][j] for j in range(n)) == 1, 'row{}'.format(i)

        # one per column
        for j in range(n):
            queens += xsum(x[i][j] for i in range(n)) == 1, 'col{}'.format(j)

        # diagonal \
        for p, k in enumerate(range(2 - n, n - 2 + 1)):
            queens += xsum(x[i][j] for i in range(n) for j in range(n)
                           if i - j == k) <= 1, 'diag1({})'.format(p)

        # diagonal /
        for p, k in enumerate(range(3, n + n)):
            queens += xsum(x[i][j] for i in range(n) for j in range(n)
                           if i + j == k) <= 1, 'diag2({})'.format(p)

        return queens

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_constr_get_index(self, solver):
        model = self.build_model(solver)

        idx = model.solver.constr_get_index('row0')
        assert idx >= 0

        idx = model.solver.constr_get_index('col0')
        assert idx >= 0

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_remove_constrs(self, solver):
        model = self.build_model(solver)

        idx1 = model.solver.constr_get_index('row0')
        assert idx1 >= 0

        idx2 = model.solver.constr_get_index('col0')
        assert idx2 >= 0

        model.solver.remove_constrs([idx1, idx2])
