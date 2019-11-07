"""Example of a Branch-and-Cut implementation for the Traveling Salesman
Problem. Initially a compact (weak) formulation is created. This formulation
is dinamically improved with cutting planes, sub-tour elimination inequalities,
using a CutsGenerator implementation. Cut generation is called from the
solver engine whenever a fractional solution is generated."""

from sys import argv, stdout as out, stderr as err
from typing import List, Tuple, Set
from time import time
import random as rnd
from os.path import basename
from itertools import product
from collections import defaultdict
import networkx as nx
import tsplib95
from mip import Model, xsum, BINARY, minimize, OptimizationStatus
from mip.callbacks import ConstrsGenerator, CutPool

def subtour(N: Set, out: defaultdict, node) -> List:
    """checks if node 'node' belongs to a subtour, returns
    elements in this sub-tour if true"""
    # BFS to search for disconected routes
    queue = [node]
    visited = set(queue)
    while queue:
        n = queue.pop()
        for nl in out[n]:
            if nl not in visited:
                queue.append(nl)
                visited.add(nl)

    if len(visited) != len(N):
        return [v for v in visited]
    else:
        return []


class SubTourLazyGenerator(ConstrsGenerator):
    """Generates lazy constraints. Removes sub-tours in integer solutions"""
    def generate_constrs(self, model: Model):
        r = [(v, v.x) for v in model.vars
             if v.name.startswith('x(') and v.x >= 0.99]
        mf = max(abs(v.x - round(v.x))
                 for v in model.vars if v.name.startswith('x('))
        assert mf <= 1e-4
        U = [int(v.name.split('(')[1].split(',')[0]) for v, f in r]
        V = [int(v.name.split(')')[0].split(',')[1]) for v, f in r]
        N, cp = set(U+V), CutPool()
        # output nodes for each node
        out = defaultdict(lambda: list())
        for i in range(len(U)):
            out[U[i]].append(V[i])

        for n in N:
            S = set(subtour(N, out, n))
            if S:
                arcsInS = [(v, f) for i, (v, f) in enumerate(r)
                           if U[i] in S and V[i] in S]
                if sum(f for v, f in arcsInS) >= (len(S)-1)+1e-4:
                    cut = xsum(1.0*v for v, fm in arcsInS) <= len(S)-1
                    cp.add(cut)
        for cut in cp.cuts:
            model += cut
        print("cuts added: %d" % len(cp.cuts))


class SubTourCutGenerator(ConstrsGenerator):
    """Class to generate cutting planes. Removes sub-tours in fractional solutions"""
    def __init__(self, Fl: List[Tuple[int, int]]):
        self.F = Fl

    def generate_constrs(self, model: Model):
        G = nx.DiGraph()
        r = [(v, v.x) for v in model.vars if v.name.startswith('x(')]
        U = [int(v.name.split('(')[1].split(',')[0]) for v, f in r]
        V = [int(v.name.split(')')[0].split(',')[1]) for v, f in r]
        cp = CutPool()
        for i in range(len(U)):
            G.add_edge(U[i], V[i], capacity=r[i][1])
        for (u, v) in F:
            if u not in U or v not in V:
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
                            model += cut
                        return
        for cut in cp.cuts:
            model += cut
        return


if len(argv) < 7:
    print('usage:tsp instanceName timeLimit threads useCuts useLazy useHeur')
    exit(1)

timeLimit = int(argv[2])
threads = int(argv[3])
useCuts = int(argv[4])
useLazy = int(argv[5])
useHeur = int(argv[6])

start = time()

inst = tsplib95.load_problem(argv[1])
V = [n-1 for n in inst.get_nodes()]
n, c = len(V), [[0.0 for j in V] for i in V]
for (i, j) in product(V, V):
    if i != j:
        c[i][j] = inst.wfunc(i+1, j+1)

model = Model()

# binary variables indicating if arc (i,j) is used on the route or not
x = [[model.add_var(var_type=BINARY, name='x(%d,%d)' % (i, j)) for j in V]
     for i in V]

# continuous variable to prevent subtours: each city will have a
# different sequential id in the planned route except the first one
y = [model.add_var(name='y(%d)' % i) for i in V]

# objective function: minimize the distance
model.objective = minimize(xsum(c[i][j]*x[i][j] for i in V for j in V))

# constraint : leave each city only once
for i in V:
    model += xsum(x[i][j] for j in set(V) - {i}) == 1

# constraint : enter each city only once
for i in V:
    model += xsum(x[j][i] for j in set(V) - {i}) == 1

if not useLazy:
    # (weak) subtour elimination
    for (i, j) in set(product(set(V) - {0}, set(V) - {0})):
        model += y[i] - (n+1)*x[i][j] >= y[j]-n

# no subtours of size 2
for (i, j) in product(V, V):
    if i != j:
        model += x[i][j] + x[j][i] <= 1

# computing farthest point for each point
F = []
G = nx.DiGraph()
for (i, j) in product(V, V):
    if i != j:
        G.add_edge(i, j, weight=c[i][j])
for i in V:
    P, D = nx.dijkstra_predecessor_and_distance(G, source=i)
    DS = list(D.items())
    DS.sort(key=lambda x: x[1])
    F.append((i, DS[-1][0]))

if useCuts:
    model.cuts_generator = SubTourCutGenerator(F)
if useLazy:
    model.lazy_constrs_generator = SubTourLazyGenerator()

if useHeur:
    # running a best insertion heuristic to obtain an initial feasible
    # solution: test every node j not yet inserted in the route at every
    # intermediate position p and select the pair (j, p) that results in the
    # smallest cost increase
    seq = [0, max((c[0][j], j) for j in V)[1]] + [0]
    Vout = set(V)-set(seq)
    while Vout:
        (j, p) = min([(c[seq[p]][j] + c[j][seq[p+1]], (j, p)) for j, p in
                      product(Vout, range(len(seq)-1))])[1]

        seq = seq[:p+1]+[j]+seq[p+1:]
        assert(seq[-1] == 0)
        Vout = Vout - {j}
    cost = sum(c[seq[i]][seq[i+1]] for i in range(len(seq)-1))
    print('route with cost %g built' % cost)

    # function to evaluate the cost of swapping two positions in a route in
    # constant time
    def delta(d: List[List[float]], S: List[int], p1: int, p2: int) -> float:
        p1, p2 = min(p1, p2), max(p1, p2)
        e1, e2 = S[p1], S[p2]
        if p1 == p2:
            return 0
        elif abs(p1-p2) == 1:
            v1 = d[S[p1-1]][e1] + d[e1][e2] + d[e2][S[p2+1]]
            v2 = d[S[p1-1]][e2] + d[e2][e1] + d[e1][S[p2+1]]
        else:
            v1 = d[S[p1-1]][e1] + d[e1][S[p1+1]] + d[S[p2-1]][e2]\
                + d[e2][S[p2+1]]
            v2 = d[S[p1-1]][e2] + d[e2][S[p1+1]] + d[S[p2-1]][e1]\
                + d[e1][S[p2+1]]
        return v2 - v1

    # applying the Late Acceptance Hill Climbing
    rnd.seed(0)
    L = [cost for i in range(50)]
    sl, cur_cost, best = seq.copy(), cost, cost
    for it in range(5000000):
        (i, j) = rnd.randint(1, len(sl)-2), rnd.randint(1, len(sl)-2)
        dlt = delta(c, sl, i, j)
        if cur_cost + dlt <= L[it % len(L)]:
            sl[i], sl[j], cur_cost = sl[j], sl[i], cur_cost + dlt
            if cur_cost < best:
                seq, best = sl.copy(), cur_cost
        L[it % len(L)] = cur_cost

    print('improved cost %g' % best)

    model.start = [(x[seq[i]][seq[i+1]], 1) for i in range(len(seq)-1)]

model.max_seconds = timeLimit
model.threads = threads
model.optimize()
end = time()

print(model.status)

print(model.solver_name)

objv = 1e20
gap = 1e20

n_nodes = 0
if model.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
    out.write('route with total distance %g found: 0'
              % (model.objective_value))
    nc = 0
    while True:
        n_nodes += 1
        nc = [i for i in V if x[nc][i].x >= 0.99][0]
        out.write(' -> %s' % nc)
        if nc == 0:
            break
    out.write('\n')
    objv = model.objective_value
    gap = model.gap

    if n_nodes != n:
        err.write('incomplete route (%d from %d) generated.\n' % (n_nodes, n))
        exit(1)

f = open('results-{}.csv'.format(basename(argv[1])), 'a')
f.write('%s,%s,%d,%d,%d,%g,%g,%g\n' % (basename(argv[1]), model.solver_name,
                                       objv, useCuts, useLazy, useHeur, gap,
                                       end-start))
f.close()
