"""Example that loads a distance matrix and solves the Traveling
   Salesman Problem using the simple compact formulation
   presented in Miller, C.E., Tucker, A.W and Zemlin, R.A. "Integer
   Programming Formulation of Traveling Salesman Problems". Journal
   of the ACM 7(4). 1960.
"""

from sys import argv
from tspdata import TSPData
from mip.model import Model, xsum
from mip.constants import BINARY

if len(argv) <= 1:
    print('enter instance name.')
    exit(1)

inst = TSPData(argv[1])
(n, d) = (inst.n, inst.d)
print('solving TSP with {} cities'.format(inst.n))

model = Model()

# binary variables indicating if arc (i,j) is used on the route or not
x = [[model.add_var(
    name='x({},{})'.format(i, j), var_type=BINARY)
      for j in range(n)] for i in range(n)]

# continuous variable to prevent subtours: each
# city will have a different "identifier" in the planned route
y = [model.add_var(name='y({})') for i in range(n)]

# objective function: minimize the distance
model.objective = xsum(d[i][j]*x[i][j]
                       for j in range(n) for i in range(n))

# constraint : enter each city coming from another city
for i in range(n):
    model += xsum(x[j][i] for j in range(n) if j != i) == 1

# constraint : leave each city coming from another city
for i in range(n):
    model += xsum(x[i][j] for j in range(n) if j != i) == 1

# subtour elimination
for i in range(1, n):
    for j in [x for x in range(1, n) if x != i]:
        model += \
            y[i] - (n+1)*x[i][j] >= y[j]-n, 'noSub({},{})'.format(i, j)

print('model has {} variables, {} of which are integral and {} rows'
      .format(model.num_cols, model.num_int, model.num_rows))

model.store_search_progress_log = True
st = model.optimize(max_seconds=20)

print('best route found has length {}, best possible (obj bound is) {} st: {}'
      .format(model.objective_value, model.objective_bound, st))

arcs = [(i, j) for i in range(n) for j in range(n) if x[i][j].x >= 0.99]
print('optimal route : {}'.format(arcs))

model.plot_bounds_evolution()
