from pulp import *
from sys import stdout
from time import process_time

n = 100

queens = LpProblem('queens', LpMinimize)

x = [[LpVariable('x({},{})'.format(i, j), 0, 1, 'Binary')
      for j in range(n)] for i in range(n)]

# objective function
queens += sum(-x[i][j] for i in range(n) for j in range(n))

# one per row
for i in range(n):
    queens += sum(x[i][j] for j in range(n)) == 1, 'row({})'.format(i)

# one per column
for j in range(n):
    queens += sum(x[i][j] for i in range(n)) == 1, 'col({})'.format(j)

# diagonal \
for p, k in enumerate(range(2 - n, n - 2 + 1)):
    queens += sum(x[i][j] for i in range(n) for j in range(n) if i - j == k) <= 1, 'diag1({})'.format(p)

# diagonal /
for p, k in enumerate(range(3, n + n)):
    queens += sum(x[i][j] for i in range(n) for j in range(n) if i + j == k) <= 1, 'diag2({})'.format(p)

queens.solve()

for i in range(n):
    for j in range(n):
        if x[i][j].varValue >= 0.98:
            stdout.write(' O')
        else:
            stdout.write(' .')
    sys.stdout.write('\n')

stdout.write('\n')
stdout.write('Total process time: {:.3f}s\n'.format(process_time()))
