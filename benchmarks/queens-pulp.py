from pulp import *
from sys import stdout
from time import process_time
import time
import timeout_decorator

N = range(100, 1001, 100)

f = open('queens-pulp.csv', 'w')

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
    queens = LpProblem('queens', LpMinimize)

    x = [[LpVariable('x({},{})'.format(i, j), 0, 1, 'Binary')
          for j in range(n)] for i in range(n)]

    # one per row
    for i in range(n):
        queens += lpSum(x[i][j] for j in range(n)) == 1, 'row({})'.format(i)

    # one per column
    for j in range(n):
        queens += lpSum(x[i][j] for i in range(n)) == 1, 'col({})'.format(j)

    # diagonal \
    for p, k in enumerate(range(2 - n, n - 2 + 1)):
        queens += lpSum(x[i][j] for i in range(n) for j in range(n)
                        if i - j == k) <= 1, 'diag1({})'.format(p)

    # diagonal /
    for p, k in enumerate(range(3, n + n)):
        queens += lpSum(x[i][j] for i in range(n) for j in range(n)
                        if i + j == k) <= 1, 'diag2({})'.format(p)

    ed = time.time()
    execTime = ed-st
    modelCols = queens.numVariables()
    modelRows = queens.numConstraints()


for n in N:
    gen_model(n)
    f.write('{},{},{:.4f}\n'.format(n, 'pulp', execTime))
    f.flush()

f.close()
