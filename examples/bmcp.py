"""Bandwidth multi coloring example
   Frequency assignment problem as described here:
   http://fap.zib.de/problems/Philadelphia/

   to solve P1 instance (included in the examples) call python bmcp.py P1.col
"""

from itertools import product
import bmcp_data
import bmcp_greedy
from mip.model import Model, xsum
from mip.constants import MINIMIZE, BINARY

data = bmcp_data.read('P1.col')
N, r, d = data.N, data.r, data.d
S = bmcp_greedy.build(data)
C, U = S.C, [i for i in range(S.u_max+1)]

m = Model(sense=MINIMIZE)

x = [[m.add_var('x({},{})'.format(i, c), var_type=BINARY)
      for c in U] for i in N]

m.objective = z = m.add_var('z')

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

m.start = [(x[i][c], 1.0) for i in N for c in C[i]]

m.optimize(max_seconds=100)

C = [[c for c in U if x[i][c] >= 0.99] for i in N]
print(C)
