"""Bandwidth multi coloring example
   Frequency assignment problem as described here:
   http://fap.zib.de/problems/Philadelphia/

   to solve P1 instance (included in the examples) call python bmcp.py P1.col
"""

from os.path import basename
from itertools import product
from sys import argv
from time import time
import bmcp_data
from mip import Model, xsum, minimize, MINIMIZE, BINARY

data = bmcp_data.read(argv[1])
ncol = int(argv[3])
N, r, d = data.N, data.r, data.d
U = list(range(ncol))

st = time()
m = Model(solver_name=argv[2])

x = [
    [m.add_var("x({},{})".format(i, c), var_type=BINARY) for c in U] for i in N
]

z = m.add_var("z")
m.objective = minimize(z)

for i in N:
    m += xsum(x[i][c] for c in U) == r[i]

for i, j, c1, c2 in product(N, N, U, U):
    if i != j and c1 <= c2 < c1 + d[i][j]:
        m += x[i][c1] + x[j][c2] <= 1

for i, c1, c2 in product(N, U, U):
    if c1 < c2 < c1 + d[i][i]:
        m += x[i][c1] + x[i][c2] <= 1

for i, c in product(N, U):
    m += z >= (c + 1) * x[i][c]
ed = time()

inst = basename(argv[1])
print("RRR:%s,%g" % (inst, ed - st))
# m.start = [(x[i][c], 1.0) for i in N for c in C[i]]

# m.optimize(max_seconds=100)

# C = [[c for c in U if x[i][c] >= 0.99] for i in N]
# print(C)
