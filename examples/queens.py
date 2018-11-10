from milppy.model import *

n = 100

queens = Model()
x = [[
	queens.add_var(
	1,
	0.0,
	1.0,
	None,
	'B',
	'x({},{})'.format(i,j) )
		for j in range(n)] for i in range(n) ]

# one per row
for i in range(n):
	queens.add_constr(
		sum(x[i][j] for j in range(n))  == 1, 'row({})'.format(i)
		)

# one per column
for j in range(n):
	queens.add_constr(
		sum(x[i][j] for i in range(n))  == 1, 'col({})'.format(j)
		)

# diagonal \
for p,k in enumerate(range(2-n,n-2+1)):
	queens.add_constr( sum(x[i][j] for i in range(n) for j in range(n) if i-j==k) <= 1, 'diag1({})'.format(p) )

# diagonal /
for p,k in enumerate(range(3,n+n)):
	queens.add_constr( sum(x[i][j] for i in range(n) for j in range(n) if i+j==k) <= 1, 'diag2({})'.format(p) )

#queens.write('queens.lp')


