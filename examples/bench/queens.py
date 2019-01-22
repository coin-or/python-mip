from mip.model import *
from sys import stdout, argv
from time import process_time
import time
import timeout_decorator

Solvers=['cbc', 'gurobi']
N = range(100,1001,100)

@timeout_decorator.timeout(1000)
def gen_model(n, solver, f):
	st = time.time()
	queens = Model('queens', MINIMIZE, solver_name=solver)

	x = [[queens.add_var('x({},{})'.format(i, j), type='B', obj=-1.0)
		  for j in range(n)] for i in range(n)]

	# one per row
	for i in range(n):
		queens += xsum(x[i][j] for j in range(n)) == 1, 'row({})'.format(i)

	# one per column
	for j in range(n):
		queens += xsum(x[i][j] for i in range(n)) == 1, 'col({})'.format(j)

	# diagonal \
	for p, k in enumerate(range(2 - n, n - 2 + 1)):
		queens += xsum(x[i][j] for i in range(n) for j in range(n) if i - j == k) <= 1, 'diag1({})'.format(p)

	# diagonal /
	for p, k in enumerate(range(3, n + n)):
		queens += xsum(x[i][j] for i in range(n) for j in range(n) if i + j == k) <= 1, 'diag2({})'.format(p)

	ed = time.time()

	f.write('{},{},{},{:.4f}\n'.format(n, queens.num_cols, queens.num_rows, ed-st))
	f.flush()


for solver in Solvers:
	f = open('queens-{}.csv'.format(solver), 'w')
	for n in N:
		gen_model(n, solver, f)
	f.close()
