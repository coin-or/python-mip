"""Example of a Branch-and-Cut implementation for the Traveling Salesman
Problem. Initially a compact (weak) formulation is created. This formulation
is dinamically improved with cutting planes, sub-tour elimination inequalities,
using a CutsGenerator implementation. Cut generation is called from the
solver engine whenever a fractional solution is generated."""

from sys import argv
from typing import List, Tuple
import networkx as nx
from tspdata import TSPData
from mip.model import Model, xsum, BINARY
from mip.callbacks import CutsGenerator, CutPool


class SubTourCutGenerator(CutsGenerator):
    """Class to generate cutting planes for the TSP"""
    def __init__(self, Fl: List[Tuple[int, int]]):
        self.F = Fl

    def generate_cuts(self, model: Model):
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
                            model.add_cut(cut)
                        return
        for cut in cp.cuts:
            model.add_cut(cut)
        return


if len(argv) <= 1:
    print('enter instance name.')
    exit(1)

inst = TSPData(argv[1])
(n, d) = (inst.n, inst.d)
print('solving TSP with {} cities'.format(inst.n))

m = Model()
# binary variables indicating if arc (i,j) is used on the route or not
x = [[m.add_var(name='x({},{})'.format(i, j),
                var_type=BINARY) for j in range(n)] for i in range(n)]

# continuous variable to prevent subtours: each
# city will have a different "identifier" in the planned route
y = [m.add_var(name='y({})'.format(i), lb=0.0, ub=n)
     for i in range(n)]

# objective function: minimize the distance
m += xsum(d[i][j] * x[i][j] for j in range(n) for i in range(n))

# constraint : enter each city coming from another city
for i in range(n):
    m += xsum(x[j][i] for j in range(n) if j != i) == 1, 'in({})'.format(i)

# constraint : leave each city coming from another city
for i in range(n):
    m += xsum(x[i][j] for j in range(n) if j != i) == 1, 'out({})'.format(i)

# no 2 subtours
for i in range(n):
    for j in [k for k in range(n) if k != i]:
        m += x[i][j] + x[j][i] <= 1

# subtour elimination weak constraint, included in the
# initial formulation to start with a complete formulation
for i in range(1, n):
    for j in [k for k in range(1, n) if k != i]:
        m += y[i] - (n + 1) * x[i][j] >= y[j] - n, 'noSub({},{})'.format(i, j)

# computing farthest point for each point
F = []
for i in range(n):
    (md, dp) = (0, -1)
    for j in [k for k in range(n) if k != i]:
        if d[i][j] > md:
            (md, dp) = (d[i][j], j)
    F.append((i, dp))

m.cuts_generator = SubTourCutGenerator(F)
m.optimize()

print('best route found has length {}'.format(m.objective_value))

for i in range(n):
    for j in [k for k in range(n) if x[i][j].x >= 0.99]:
        print('arc({},{})'.format(i, j))
