from tspdata import TSPData
from sys import argv
from mip.model import *
from typing import List
from math import inf


# simple heuristic to generate an initial feasible solution
def gen_ini_sol(n: int, d: List[List[float]]) -> List[int]:
    sol = [0, n-1]
    out = set(range(1, n-1))
    # performing cheapest insertion
    while len(out):
        (bestDelta, bestK, bestPoint) = (inf, None, None)
        for newPoint in out:
            for k in range(len(sol)):
                delta = d[sol[k-1]][newPoint] + d[newPoint][sol[k]] - \
                    d[sol[k-1]][sol[k]]
                if delta < bestDelta:
                    bestPoint = newPoint
                    bestDelta = delta
                    bestK = k

        out.remove(bestPoint)
        sol.insert(bestK, bestPoint)

    dist = 0.0
    for k in range(len(sol)):
        dist += d[sol[k-1]][sol[k]]

    print('route with length {} found by heuristic'.format(dist))

    return sol


if len(argv) <= 1:
    print('enter instance name.')
    exit(1)

inst = TSPData(argv[1])
n = inst.n
d = inst.d
print('solving TSP with {} cities'.format(inst.n))

m = Model()

# binary variables indicating if arc (i,j) is used on the route or not
x = [[m.add_var(name='x({},{})'.format(i, j), var_type=BINARY)
      for j in range(n)] for i in range(n)]

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

# adding heuristic initial solution
hsol = gen_ini_sol(n, d)

m.start = [(x[hsol[k-1]][hsol[k]], 1.0) for k in range(n)]

m.optimize(max_seconds=60)

print('best route found has length {}'.format(m.objective_value))

for i in range(n):
    for j in range(n):
        if x[i][j].x >= 0.98:
            print('arc ({},{})'.format(i, j))
