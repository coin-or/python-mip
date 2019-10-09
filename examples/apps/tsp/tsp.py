"""Example of a Branch-and-Cut implementation for the Traveling Salesman
Problem. Initially a compact (weak) formulation is created. This formulation
is dinamically improved with cutting planes, sub-tour elimination inequalities,
using a CutsGenerator implementation. Cut generation is called from the
solver engine whenever a fractional solution is generated."""

from collections import defaultdict
from sys import argv
from typing import List, Tuple
from time import time
from os.path import basename
from itertools import product
import networkx as nx
import tsplib95
from mip.model import Model, xsum, BINARY, minimize
from mip.callbacks import ConstrsGenerator, CutPool


class SubTourCutGenerator(ConstrsGenerator):
    """Class to generate cutting planes for the TSP"""
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
N = [n for n in inst.get_nodes()]
n = len(N)
A = dict()
for (i, j) in inst.get_edges():
    if i != j:
        A[(i, j)] = inst.wfunc(i, j)

# set of edges leaving a node
OUT = defaultdict(set)

# set of edges entering a node
IN = defaultdict(set)

# an arbitrary initial point
n0 = min(i for i in N)

for a in A:
    OUT[a[0]].add(a)
    IN[a[1]].add(a)

print('solving TSP with {} cities'.format(len(N)))

model = Model()

# binary variables indicating if arc (i,j) is used on the route or not
x = {a: model.add_var('x({},{})'.format(a[0], a[1]), var_type=BINARY) for a in A}

# continuous variable to prevent subtours: each
# city will have a different "identifier" in the planned route
y = {i: model.add_var(name='y({})') for i in N}

# continuous variable to prevent subtours: each
# city will have a different "identifier" in the planned route
y = {i: model.add_var(name='y({})') for i in N}

# objective function: minimize the distance
model.objective = minimize(xsum(A[a]*x[a] for a in A))

# constraint : enter each city coming from another city
for i in N:
    model += xsum(x[a] for a in OUT[i]) == 1

# constraint : leave each city coming from another city
for i in N:
    model += xsum(x[a] for a in IN[i]) == 1

if not useLazy:
    # (weak) subtour elimination
    for (i, j) in [a for a in A if n0 not in [a[0], a[1]]]:
        model += \
            y[i] - (n+1)*x[(i, j)] >= y[j]-n, 'noSub({},{})'.format(i, j)

# no subtours of size 2
for a in A:
    if (a[1], a[0]) in A.keys():
        model += x[a] + x[a[1], a[0]] <= 1

# computing farthest point for each point
F = []
G = nx.DiGraph()
for ((i, j), d) in A.items():
    G.add_edge(i, j, weight=d)
for i in N:
    P, D = nx.dijkstra_predecessor_and_distance(G, source=i)
    DS = list(D.items())
    DS.sort(key=lambda x: x[1])
    F.append((i, DS[-1][0]))

if useCuts:
    model.cuts_generator = SubTourCutGenerator(F)
if useLazy:
    model.lazy_constrs_generator = SubTourCutGenerator(F)

if useHeur:
    seq = [n0, max((A[n0, a[1]], a[1]) for a in A if a[0] == n0)[1]]
    Vout = set(N)-set(seq)
    while Vout:
        L = [(A[(seq[p], nl)]+A[(nl, seq[p+1])], (nl, p))
             for (nl, p) in product(Vout, range(len(seq)-1))
             if (seq[p], nl) in A and (nl, seq[p+1]) in A]
        if not L:
            print('initial tour not found by heuristic')
            break
        LM = min(L)
        nn, p = LM[1]
        seq = seq[:p+1]+[nn]+seq[p+1:]
        Vout = Vout - {nn}
    
    if len(seq) == n:
        model.start = [(x[(seq[p], seq[p+1])], 1.0) for p in range(len(seq)-1)]    
    
model.max_seconds = timeLimit
model.threads = threads
model.optimize()
end = time()

print(model.status)

print(model.solver_name)

objv = 1e20
gap = 1e20
if model.num_solutions:
    print('best route found has length {}'.format(model.objective_value))
    arcs = [a for a in A.keys() if x[a].x >= 0.99]
    print('optimal route : {}'.format(arcs))
    objv = model.objective_value
    gap = model.gap

f=open('results.csv', 'a')
f.write('%s,%s,%d,%d,%d,%g,%g,%g\n' % (basename(argv[1]), model.solver_name, objv, useCuts, useLazy, useHeur, gap, end-start))
f.close()
