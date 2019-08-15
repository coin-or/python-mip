"""Example of a solver to the n-queens problem:
   n chess queens should be placed in a n x n
   chess board so that no queen can attack another,
   i.e., just one queen per line, column and diagonal.
"""

from sys import stdout
from mip.model import Model, xsum
from mip.constants import MAXIMIZE, BINARY
from mip.callbacks import CutsGenerator


class DiagonalCutGenerator(CutsGenerator):

    def generate_cuts(self, model: Model):
        def row(vname: str) -> str:
            return int(vname.split('(')[1].split(',')[0].split(')')[0])

        def col(vname: str) -> str:
            return int(vname.split('(')[1].split(',')[1].split(')')[0])

        x = {(row(v.name), col(v.name)): v for v in model.vars}
        for p, k in enumerate(range(2 - n, n - 2 + 1)):
            cut = xsum(x[i, j] for i in range(n) for j in range(n)
                       if i - j == k) <= 1
            if cut.violation > 0.001:
                model.add_cut(cut)

        for p, k in enumerate(range(3, n + n)):
            cut = xsum(x[i, j] for i in range(n) for j in range(n)
                       if i + j == k) <= 1
            if cut.violation > 0.001:
                model.add_cut(cut)


# number of queens
n = 8

queens = Model('queens', MAXIMIZE)

x = [[queens.add_var('x({},{})'.format(i, j), var_type=BINARY)
      for j in range(n)] for i in range(n)]

# one per row
for i in range(n):
    queens += xsum(x[i][j] for j in range(n)) == 1, 'row({})'.format(i)

# one per column
for j in range(n):
    queens += xsum(x[i][j] for i in range(n)) == 1, 'col({})'.format(j)


queens.cuts_generator = DiagonalCutGenerator()
queens.cuts_generator.lazy_constraints = True
queens.optimize()

stdout.write('\n')
for i, v in enumerate(queens.vars):
    stdout.write('O ' if v.x >= 0.99 else '. ')
    if i % n == n-1:
        stdout.write('\n')
