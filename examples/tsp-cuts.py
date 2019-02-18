from tspdata import TSPData
from sys import argv
from mip.model import *
from mip.constants import *
import networkx as nx
from math import floor


class SubTourCutGenerator(CutsGenerator):
    def __init__(self, model: "Model", n: int):
        super().__init__(model)
        self.n = n

    def generate_cuts(self, relax_solution: List[Tuple[Var, float]]) -> List[LinExpr]:
        G = nx.DiGraph()
        for i, (v, x) in enumerate(relax_solution):
            if 'x(' not in v.name:
                continue
            strarc = v.name.split('(')[1].split(')')[0]
            if abs(x) < 1e-6:
                continue
            ui = int(strarc.split(',')[0].strip())
            vi = int(strarc.split(',')[1].strip())
            G.add_edge(ui, vi, capacity=int(floor(x * 10000.0)))

        cuts = []

        for u in range(self.n):
            for v in range(self.n):
                if u == v: continue
                val, part = nx.minimum_cut(G, u, v)
                # checking violation
                if val >= 9999:
                    continue

                reachable, nonreachable = part

                cutvars = list()

                for u in reachable:
                    for v in nonreachable:
                        var = model.get_var_by_name('x({},{})'.format(u, v))
                        if var != None:
                            cutvars.append(var) 
                if len(cutvars):
                    cuts.append(xsum(1.0*var for var in cutvars) >= 1)

        #print("Cuts: ", cuts)
        return cuts


if len(argv) <= 1:
    print('enter instance name.')
    exit(1)

inst = TSPData(argv[1])
n = inst.n
d = inst.d
print('solving TSP with {} cities'.format(inst.n))

model = Model()

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
