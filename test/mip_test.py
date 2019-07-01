"""Tests of Python-MIP framework, should be executed before each new
release"""
from mip.model import Model, xsum
from mip.constants import OptimizationStatus, MAXIMIZE, BINARY, INTEGER

SOLVERS = ['cbc', 'gurobi']


def test_queens(solver: str):
    """MIP model n-queens"""
    n = 50
    print("> n-queens test solver engine {}".format(solver))

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

    assert queens.status == OptimizationStatus.OPTIMAL

    # querying problem variables and checking opt results
    total_queens = 0
    for v in queens.vars:
        # basic integrality test
        assert v.x <= 0.0001 or v.x >= 0.9999
        total_queens += v.x

    assert abs(total_queens - n) <= 0.001

    # solution feasibility
    for i in range(n):
        assert abs(sum(x[i][j].x for j in range(n))-1) <= 0.001


def test_tsp(solver: str):
    print("> tsp-root test solver engine {}".format(solver))
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

    m.objective = xsum(c*x[a] for a, c in A.items())

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
                y[i] - (n+1)*x[(i, j)] >= y[j]-n)

    m.relax()
    m.optimize()

    assert m.status == OptimizationStatus.OPTIMAL
    assert (abs(m.objective_value - 238.75)) <= 1e-4

    # setting all variables to integer now
    for v in m.vars:
        v.var_type = INTEGER

    m.optimize()

    assert m.status == OptimizationStatus.OPTIMAL
    assert (abs(m.objective_value - 262)) <= 1e-4


for svr in SOLVERS:
    test_queens(svr)
    test_tsp(svr)

print("")
print("================")
print("All test passed!")
print("================")
