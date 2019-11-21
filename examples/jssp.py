"""Job Shop Scheduling Problem Python-MIP example
   To execute it on the example instance ft03.jssp call
   python jssp.py ft03.jssp
   by Victor Silva"""

from itertools import product
from mip import Model, BINARY

n = m = 3

times = [[2, 1, 2],
         [1, 2, 2],
         [1, 2, 1]]

M = sum(times[i][j] for i in range(n) for j in range(m))

machines = [[2, 0, 1],
            [1, 2, 0],
            [2, 1, 0]]

model = Model('JSSP')

c = model.add_var(name="C")
x = [[model.add_var(name='x({},{})'.format(j+1, i+1))
      for i in range(m)] for j in range(n)]
y = [[[model.add_var(var_type=BINARY, name='y({},{},{})'.format(j+1, k+1, i+1))
       for i in range(m)] for k in range(n)] for j in range(n)]

model.objective = c

for (j, i) in product(range(n), range(1, m)):
    model += x[j][machines[j][i]] - x[j][machines[j][i-1]] >= \
        times[j][machines[j][i-1]]

for (j, k) in product(range(n), range(n)):
    if k != j:
        for i in range(m):
            model += x[j][i] - x[k][i] + M*y[j][k][i] >= times[k][i]
            model += -x[j][i] + x[k][i] - M*y[j][k][i] >= times[j][i] - M

for j in range(n):
    model += c - x[j][machines[j][m - 1]] >= times[j][machines[j][m - 1]]

model.optimize()

print("Completion time: ", c.x)
for (j, i) in product(range(n), range(m)):
    print("task %d starts on machine %d at time %g " % (j+1, i+1, x[j][i].x))

# sanity tests
from mip import OptimizationStatus
assert model.status == OptimizationStatus.OPTIMAL
assert round(c.x) == 7
