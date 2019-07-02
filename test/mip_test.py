"""Tests of Python-MIP framework, should be executed before each new
release"""
from itertools import product
import networkx as nx
from mip.model import Model, xsum
from mip.constants import OptimizationStatus, MAXIMIZE, BINARY, INTEGER
from mip.callbacks import CutsGenerator, CutPool
has_colors = True
try:
    from termcolor import colored
except ModuleNotFoundError:
    has_colors = False


SOLVERS = ['CBC', 'Gurobi']


def announce_test(descr: str, engine: str):
    """print test starting message"""
    if has_colors:
        test = colored('> Test: ', 'cyan') + \
            colored('{} '.format(descr), 'cyan',
                    attrs=['bold']) + \
            colored('    Solver engine: ', 'cyan') + \
            colored('{} '.format(engine), 'cyan',
                    attrs=['bold'])
    else:
        test = '> Test: {} Solver engine: {}'.format(descr, engine)

    print(test)


def check_result(descr: str, test: bool):
    """executes a test and prints the result, raising exception
    when fail"""

    if test:
        if has_colors:
            text = '\t'+colored('{}'.format(descr), 'cyan') + \
                ' : ' + colored('OK', 'green')
        else:
            text = '\t'+'{} : OK'.format(descr)

        print(text)
    else:
        print('\t{} : FAILED'.format(descr))
        raise Exception("assert failed: {}".format(descr))


def test_queens(solver: str):
    """MIP model n-queens"""
    n = 50
    announce_test("n-Queens", solver)

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

    check_result("model status", queens.status == OptimizationStatus.OPTIMAL)

    # querying problem variables and checking opt results
    total_queens = 0
    for v in queens.vars:
        # basic integrality test
        assert v.x <= 0.0001 or v.x >= 0.9999
        total_queens += v.x

    # solution feasibility
    rows_with_queens = 0
    for i in range(n):
        if abs(sum(x[i][j].x for j in range(n))-1) <= 0.001:
            rows_with_queens += 1

    check_result("feasible solution", abs(total_queens - n) <= 0.001 and
                 rows_with_queens == n)
    print('')


def test_tsp(solver: str):
    """tsp related tests"""
    announce_test("TSP - Compact", solver)
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

    check_result("lp model status", m.status == OptimizationStatus.OPTIMAL)
    check_result("lp model objective", (abs(m.objective_value - 238.75))
                 <= 1e-4)

    # setting all variables to integer now
    for v in m.vars:
        v.var_type = INTEGER

    m.optimize()

    check_result("mip model status", m.status == OptimizationStatus.OPTIMAL)
    check_result("mip model objective", (abs(m.objective_value - 262)) <=
                 0.0001)
    print('')


class SubTourCutGenerator(CutsGenerator):
    """Class to generate cutting planes for the TSP"""

    def generate_cuts(self, model: Model):
        G = nx.DiGraph()
        r = [(v, v.x) for v in model.vars if v.name.startswith('x(')]
        U = [v.name.split('(')[1].split(',')[0] for v, f in r]
        V = [v.name.split(')')[0].split(',')[1] for v, f in r]
        N = list(set(U+V))
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
                if sum(f for v, f in arcsInS) >= (len(S)-1)+1e-4:
                    cut = xsum(1.0*v for v, fm in arcsInS) <= len(S)-1
                    cp.add(cut)
                    if len(cp.cuts) > 256:
                        for cut in cp.cuts:
                            model.add_cut(cut)
                        return
        for cut in cp.cuts:
            model.add_cut(cut)
        return


def test_tsp_cuts(solver: str):
    """tsp related tests"""
    announce_test("TSP - Branch & Cut", solver)
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

    m.cuts_generator = SubTourCutGenerator()

    # tiny model, should be enough to find the optimal
    m.max_seconds = 10
    m.max_nodes = 100
    m.max_solutions = 1000

    m.optimize()

    check_result("mip model status", m.status == OptimizationStatus.OPTIMAL)
    check_result("mip model objective", (abs(m.objective_value - 262)) <=
                 0.0001)
    print('')


def test_tsp_mipstart(solver: str):
    """tsp related tests"""
    announce_test("TSP - MIPStart", solver)
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

    route = ['a', 'g', 'f', 'c', 'd', 'e', 'b', 'a']
    m.start = [(x[route[i-1], route[i]], 1.0) for i in range(1, len(route))]
    m.optimize()

    check_result("mip model status", m.status == OptimizationStatus.OPTIMAL)
    check_result("mip model objective", (abs(m.objective_value - 262)) <=
                 0.0001)
    print('')


print("")
if has_colors:
    print(colored("=======================================",
                  "green", attrs=["bold"]))
    print("")
    print(colored("      Starting Automated Tests",
                  "green", attrs=["bold"]))
    print("")
    print(colored("=======================================",
                  "green", attrs=["bold"]))
else:
    print("=======================================")
    print("")
    print("       Starting Automated Tests")
    print("")
    print("=======================================")
for svr in SOLVERS:
    test_queens(svr)
    test_tsp(svr)
    test_tsp_cuts(svr)
    test_tsp_mipstart(svr)

print("")
if has_colors:
    print(colored("=======================================",
                  "green", attrs=["bold"]))
    print("")
    print(colored("            All test passed", "green", attrs=["bold"]))
    print(colored("                   :^)", "green", attrs=["bold"]))
    print("")
    print(colored("=======================================",
                  "green", attrs=["bold"]))
else:
    print("=======================================")
    print("")
    print("            All test passed")
    print("                  :^)")
    print("")
    print("=======================================")
