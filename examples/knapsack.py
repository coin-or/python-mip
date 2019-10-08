"""0/1 Knapsack example"""

from mip import Model, xsum, maximize, BINARY

p = [10, 13, 18, 31, 7, 15]
w = [11, 15, 20, 35, 10, 33]
c = 47
n = len(w)

m = Model('knapsack')

x = [m.add_var(var_type=BINARY) for i in range(n)]

m.objective = maximize(xsum(p[i] * x[i] for i in range(n)))

m += xsum(w[i] * x[i] for i in range(n)) <= c

m.optimize()

selected = [i for i in range(n) if x[i].x >= 0.99]
print('selected items: {}'.format(selected))
