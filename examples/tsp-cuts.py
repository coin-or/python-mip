"""Example of a Branch-and-Cut implementation for the Traveling Salesman
Problem. Initially a compact (weak) formulation is created. This formulation is
dynamically improved with sub-tour elimination inequalities, using a
CutsGenerator implementation that repeatedly solves the min-cut problem between
pairs of distant nodes to find isolated sets of nodes. Cut generation is called
from the solver engine whenever a fractional solution is generated."""

from typing import List, Tuple
from random import seed, randint
from itertools import product
from math import sqrt
import networkx as nx
from mip import Model, xsum, BINARY, minimize, ConstrsGenerator, CutPool


class SubTourCutGenerator(ConstrsGenerator):
    """Class to generate cutting planes for the TSP"""
    def __init__(self, Fl: List[Tuple[int, int]], x_, V_):
        self.F, self.x, self.V = Fl, x_, V_

    def generate_constrs(self, model: Model, depth: int = 0, npass: int = 0):
        xf, V_, cp, G = model.translate(self.x), self.V, CutPool(), nx.DiGraph()
        for (u, v) in [(k, l) for (k, l) in product(V_, V_) if k != l and xf[k][l]]:
            G.add_edge(u, v, capacity=xf[u][v].x)
        for (u, v) in F:
            val, (S, NS) = nx.minimum_cut(G, u, v)
            if val <= 0.99:
                aInS = [(xf[i][j], xf[i][j].x)
                        for (i, j) in product(V_, V_) if i != j and xf[i][j] and i in S and j in S]
                if sum(f for v, f in aInS) >= (len(S)-1)+1e-4:
                    cut = xsum(1.0*v for v, fm in aInS) <= len(S)-1
                    cp.add(cut)
                    if len(cp.cuts) > 256:
                        for cut in cp.cuts:
                            model += cut
                        return
        for cut in cp.cuts:
            model += cut


n = 30  # number of points
V = set(range(n))
seed(0)
p = [(randint(1, 100), randint(1, 100)) for i in V]  # coordinates
Arcs = [(i, j) for (i, j) in product(V, V) if i != j]

# distance matrix
c = [[round(sqrt((p[i][0]-p[j][0])**2 + (p[i][1]-p[j][1])**2)) for j in V] for i in V]

model = Model()

# binary variables indicating if arc (i,j) is used on the route or not
x = [[model.add_var(var_type=BINARY) for j in V] for i in V]

# continuous variable to prevent subtours: each city will have a
# different sequential id in the planned route except the first one
y = [model.add_var() for i in V]

# objective function: minimize the distance
model.objective = minimize(xsum(c[i][j]*x[i][j] for (i, j) in Arcs))

# constraint : leave each city only once
for i in V:
    model += xsum(x[i][j] for j in V - {i}) == 1

# constraint : enter each city only once
for i in V:
    model += xsum(x[j][i] for j in V - {i}) == 1

# (weak) subtour elimination
# subtour elimination
for (i, j) in product(V - {0}, V - {0}):
    if i != j:
        model += y[i] - (n+1)*x[i][j] >= y[j]-n

# no subtours of size 2
for (i, j) in Arcs:
    model += x[i][j] + x[j][i] <= 1

# computing farthest point for each point, these will be checked first for
# isolated subtours
F, G = [], nx.DiGraph()
for (i, j) in Arcs:
    G.add_edge(i, j, weight=c[i][j])
for i in V:
    P, D = nx.dijkstra_predecessor_and_distance(G, source=i)
    DS = list(D.items())
    DS.sort(key=lambda x: x[1])
    F.append((i, DS[-1][0]))

model.cuts_generator = SubTourCutGenerator(F, x, V)
model.optimize()

print('status: %s route length %g' % (model.status, model.objective_value))

# sanity check
model.check_optimization_results()
