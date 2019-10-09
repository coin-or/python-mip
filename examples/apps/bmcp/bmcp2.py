"""Bandwidth multi coloring example
   Frequency assignment problem as described here:
   http://fap.zib.de/problems/Philadelphia/

   to solve P1 instance (included in the examples) call python bmcp.py P1.col
"""

from itertools import product
from mip import Model, xsum, minimize, BINARY

# number of channels per node
r = [3, 5, 8, 3, 6, 5, 7, 3]

# distance between channels in the same node (i, i) and in adjacent nodes
#      0  1  2  3  4  5  6  7
d = [[3, 2, 0, 0, 2, 2, 0, 0],   # 0
     [2, 3, 2, 0, 0, 2, 2, 0],   # 1
     [0, 2, 3, 0, 0, 0, 3, 0],   # 2
     [0, 0, 0, 3, 2, 0, 0, 2],   # 3
     [2, 0, 0, 2, 3, 2, 0, 0],   # 4
     [2, 2, 0, 0, 2, 3, 2, 0],   # 5
     [0, 2, 2, 0, 0, 2, 3, 0],   # 6
     [0, 0, 0, 2, 0, 0, 0, 3]]   # 7

N = range(len(r))

# in complete applications this upper bound should be obtained from a feasible
# solution produced with some heuristic
U = range(sum(d[i][j] for (i, j) in product(N, N)) + sum(el for el in r))

m = Model()

x = [[m.add_var('x({},{})'.format(i, c), var_type=BINARY)
      for c in U] for i in N]

z = m.add_var('z')
m.objective = minimize(z)

for i in N:
    m += xsum(x[i][c] for c in U) == r[i]

for i, j, c1, c2 in product(N, N, U, U):
    if i != j and c1 <= c2 < c1+d[i][j]:
        m += x[i][c1] + x[j][c2] <= 1

for i, c1, c2 in product(N, U, U):
    if c1 < c2 < c1+d[i][i]:
        m += x[i][c1] + x[i][c2] <= 1

for i, c in product(N, U):
    m += z >= (c+1)*x[i][c]

m.optimize(max_seconds=100)

if m.num_solutions:
    for i in N:
        print('Channels of node %d: %s', i, [c for c in U if x[i][c] >= 0.99])
