from tspdata import TSPData
from sys import argv
from mip.model import *
from mip.constants import *
import networkx as nx
from math import floor
from itertools import product

class SubTourCutGenerator(CutsGenerator):
    """Class to separate sub-tour elimination constraints"""
    def __init__(self, model: Model, n: int):
        super().__init__(model)
        self.n = n

    def generate_cuts(self, relax_solution: List[Tuple[Var, float]]) -> List[LinExpr]:
        """assumes that variable names have the format x(i,j)"""
        G = nx.DiGraph()
        # only x variables
        r = [(v,f) for (v,f) in relax_solution if 'x(' in v.name]
        # getting tails and heads of arcs
        U = [int(v.name.split('(')[1].split(',')[0]) for v,f in r]
        V = [int(v.name.split(')')[0].split(',')[1]) for v,f in r]
        for i in range(len(U)):
            G.add_edge(U[i], V[i], capacity=r[i][1])
        cuts = []
        for (u,v) in [(u,v) for (u,v) in product(range(n),range(n)) if u!=v]:
            val, (S,NS) = nx.minimum_cut(G, u, v)
            if min(len(S), len(NS))<=2:
                continue
            # checking violation
            if val > 0.99:
                continue
            arcsInS = [(v,f) for i,(v,f) in enumerate(r) if U[i] in S and V[i] in S]
            sumArcsInS = sum(f for v,f in arcsInS)
            if sumArcsInS > len(S)-1:
                cut = xsum(1.0*v for v,fm in arcsInS) <= len(S)-1
                print(cut)
                cuts.append(cut)

        return cuts


if len(argv) <= 1:
    print('enter instance name.')
    exit(1)

inst = TSPData(argv[1])
n = inst.n
d = inst.d
print('solving TSP with {} cities'.format(inst.n))

model = Model()
model.threads = 1

# binary variables indicating if arc (i,j) is used on the route or not
x = [[model.add_var(
    name='x({},{})'.format(i, j),
    var_type=BINARY)
    for j in range(n)]
    for i in range(n)]

# continuous variable to prevent subtours: each
# city will have a different "identifier" in the planned route
y = [model.add_var(
    name='y({})'.format(i),
    lb=0.0,
    ub=n)
    for i in range(n)]

# objective function: minimize the distance
model += xsum(d[i][j] * x[i][j]
              for j in range(n) for i in range(n))

# constraint : enter each city coming from another city
for i in range(n):
    model += xsum(x[j][i] for j in range(n) if j != i) == 1, 'enter({})'.format(i)

# constraint : leave each city coming from another city
for i in range(n):
    model += xsum(x[i][j] for j in range(n) if j != i) == 1, 'leave({})'.format(i)

# no 2 subtours
for i in range(n):
    for j in range(n):
        if j != j:
            model += x[i][j] + x[j][i] <= 1

# subtour elimination
for i in range(0, n):
    for j in range(0, n):
        if i == j or i == 0 or j == 0:
            continue
        model += \
            y[i] - (n + 1) * x[i][j] >= y[j] - n, 'noSub({},{})'.format(i, j)

model.add_cut_generator(SubTourCutGenerator(model, n))
model.optimize(max_seconds=60, max_nodes=100)

print('best route found has length {}'.format(model.objective_value))

for i in range(n):
    for j in range(n):
        if x[i][j].x >= 0.98:
            print('arc ({},{})'.format(i, j))

print('finished')
