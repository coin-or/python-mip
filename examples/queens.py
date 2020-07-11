"""Example of a solver to the n-queens problem:  n chess queens should be
placed in a n x n chess board so that no queen can attack another, i.e., just
one queen per line, column and diagonal.  """

from sys import stdout
from mip import Model, xsum, BINARY

# number of queens
n = 40

queens = Model()

x = [[queens.add_var('x({},{})'.format(i, j), var_type=BINARY)
      for j in range(n)] for i in range(n)]

# one per row
for i in range(n):
    queens += xsum(x[i][j] for j in range(n)) == 1, 'row({})'.format(i)

# one per column
for j in range(n):
    queens += xsum(x[i][j] for i in range(n)) == 1, 'col({})'.format(j)

# diagonal \
for p, k in enumerate(range(2 - n, n - 2 + 1)):
    queens += xsum(x[i][i - k] for i in range(n)
                   if 0 <= i - k < n) <= 1, 'diag1({})'.format(p)

# diagonal /
for p, k in enumerate(range(3, n + n)):
    queens += xsum(x[i][k - i] for i in range(n)
                   if 0 <= k - i < n) <= 1, 'diag2({})'.format(p)

queens.optimize()

if queens.num_solutions:
    stdout.write('\n')
    for i, v in enumerate(queens.vars):
        stdout.write('O ' if v.x >= 0.99 else '. ')
        if i % n == n-1:
            stdout.write('\n')
