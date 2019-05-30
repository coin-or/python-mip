from gurobipy import *
from sys import stdout
import time
# from time import process_time
import timeout_decorator

N = range(100, 1001, 100)

TIMEOUT = 1000
execTime = TIMEOUT
modelCols = 0
modelRows = 0
modelNz = 0


@timeout_decorator.timeout(TIMEOUT)
def gen_model(n):
    global execTime
    global modelCols
    global modelRows
    global modelNz
    execTime = TIMEOUT
    modelCols = 0
    modelRows = 0
    modelNz = 0

    st = time.time()
    queens = Model('queens')

    x = [[queens.addVar(name='x({},{})'.format(i, j), vtype='B', obj=-1)
          for j in range(n)] for i in range(n)]

    # one per row
    for i in range(n):
        queens.addConstr(quicksum(x[i][j] for j in range(n)) == 1,
                         name='row({})'.format(i))

    # one per column
    for j in range(n):
        queens.addConstr(quicksum(x[i][j] for i in range(n)) == 1,
                         name='col({})'.format(j))

    # diagonal \
    for p, k in enumerate(range(2 - n, n - 2 + 1)):
        queens.addConstr(quicksum(x[i][j] for i in range(n) for j in range(n)
                                  if i - j == k) <= 1, name='diag1({})'.format(p))

    # diagonal /
    for p, k in enumerate(range(3, n + n)):
        queens.addConstr(quicksum(x[i][j] for i in range(n) for j in range(n)
                                  if i + j == k) <= 1, name='diag2({})'.format(p))

    ed = time.time()

    execTime = ed-st
    modelCols = queens.NumVars
    modelRows = queens.NumConstrs


f = open('queens-gurobi-cpython.csv', 'w')

for n in N:
    gen_model(n)
    f.write('{},{},{:.4f}\n'.format(n, 'gurobi', execTime))
    f.flush()

f.close()
