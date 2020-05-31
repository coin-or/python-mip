"""0/1 Knapsack example"""

from mip import Model, xsum, maximize, BINARY

p = [10, 13, 18, 31, 7, 15]
w = [11, 15, 20, 35, 10, 33]
c, I = 47, range(len(w))

m = Model("knapsack")

x = [m.add_var(var_type=BINARY) for i in I]

m.objective = maximize(xsum(p[i] * x[i] for i in I))

m += xsum(w[i] * x[i] for i in I) <= c

m.optimize()

selected = [i for i in I if x[i].x >= 0.99]
print("selected items: {}".format(selected))

# sanity tests
from mip import OptimizationStatus

assert m.status == OptimizationStatus.OPTIMAL
assert round(m.objective_value) == 41
assert round(m.constrs[0].slack) == 1
