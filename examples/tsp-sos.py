"""Example that loads a distance matrix and solves the Traveling
   Salesman Problem using the simple compact formulation
   presented in Miller, C.E., Tucker, A.W and Zemlin, R.A. "Integer
   Programming Formulation of Traveling Salesman Problems". Journal
   of the ACM 7(4). 1960.
"""

from collections import defaultdict
from sys import argv
import tsplib95
from mip.model import Model, xsum, minimize
from mip.constants import BINARY


if len(argv) <= 1:
    print('enter instance name.')
    exit(1)

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
x = {a: model.add_var('x({},{})'.format(a[0], a[1]), var_type=BINARY)
     for a in A.keys()}

# continuous variable to prevent subtours: each
# city will have a different "identifier" in the planned route
y = {i: model.add_var(name='y({})') for i in N}

# objective function: minimize the distance
model.objective = minimize(
    xsum(A[a]*x[a] for a in A.keys()))

# constraint : enter each city coming from another city
for i in N:
    model += xsum(x[a] for a in OUT[i]) == 1

# constraint : leave each city coming from another city
for i in N:
    model += xsum(x[a] for a in IN[i]) == 1

# subtour elimination
for (i, j) in [a for a in A.keys() if n0 not in [a[0], a[1]]]:
    model += \
        y[i] - (n+1)*x[(i, j)] >= y[j]-n, 'noSub({},{})'.format(i, j)

print('model has {} variables, {} of which are integral and {} rows'
      .format(model.num_cols, model.num_int, model.num_rows))

print("Adding SOSs")
for i in N:
    sosOut = [(x[(i, j)], A[(i, j)]) for (i, j) in OUT[i]]
    sosIn = [(x[(i, j)], A[(i, j)]) for (i, j) in IN[i]]
    model.add_sos(sosOut, 1)

model.max_nodes = 1000
st = model.optimize(max_seconds=120)

print('best route found has length {}, best possible (obj bound is) {} st: {}'
      .format(model.objective_value, model.objective_bound, st))

arcs = [(a) for a in A.keys() if x[a].x >= 0.99]
print('optimal route : {}'.format(arcs))
