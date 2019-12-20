#!/usr/bin/python
# -*- coding: utf-8 -*-

from mip import Model, xsum, BINARY, INTEGER

n = 10  # maximum number of bars
L = 250  # bar length
m = 4  # number of requests
w = [187, 119, 74, 90]  # size of each item
b = [1, 2, 2, 1]  # demand for each item

# creating the model
model = Model()
x = {(i, j): model.add_var(obj=0, var_type=INTEGER, name="x[%d,%d]" % (i, j))
     for i in range(m) for j in range(n)}
y = {j: model.add_var(obj=1, var_type=BINARY, name="y[%d]" % j)
     for j in range(n)}

# constraints
for i in range(m):
    model.add_constr(xsum(x[i, j] for j in range(n)) >= b[i])
for j in range(n):
    model.add_constr(xsum(w[i] * x[i, j] for i in range(m)) <= L * y[j])

# additional constraints to reduce symmetry
for j in range(1, n):
    model.add_constr(y[j - 1] >= y[j])

# optimizing the model
model.optimize()

# printing the solution
print('')
print('Objective value: {model.objective_value:.3}'.format(**locals()))
print('Solution: ', end='')
for v in model.vars:
    if v.x > 1e-5:
        print('{v.name} = {v.x}'.format(**locals()))
        print('          ', end='')

# sanity tests
from mip import OptimizationStatus
assert abs(model.objective_value - 3) <= 1e-4
assert sum(x.x for x in model.vars) >= 5
