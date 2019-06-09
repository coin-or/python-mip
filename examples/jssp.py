"""Job Shop Scheduling Problem Python-MIP exaxmple
   To execute it on the example instance ft03.jssp call
   python jssp.py ft03.jssp
   by Victor Silva"""

from itertools import product
from sys import argv
from jssp_instance import JSSPInstance
from mip.model import Model
from mip.constants import BINARY

inst = JSSPInstance(argv[1])
n, m, machines, times, M = inst.n, inst.m, inst.machines, inst.times, inst.M

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

print("C: ", c.x)
for j in range(n):
    for i in range(m):
        print('x({},{}) = {} '.format(j+1, i+1, x[j][i].x), end='')
    print()
