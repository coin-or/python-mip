from mip.model import *
from sys import argv
from typing import List
from sys import stdout
from math import inf

# to prevent hard to interpret cuts, set to inf 
# if you want an easier to solve model
maxMult = 5

# maximum absolute value of coefficients in cut
maxCutCoef = 10

def std_model( omip : Model ) -> Model:
	smodel = Model( name=omip.name, sense=omip.sense, solver_name=omip.solver_name )
	for v in omip.vars:
		smodel.add_var( v.name, v.lb, v.ub, v.obj, v.type )

	for c in omip.constrs:
		ce = c.expr
		a = c.name
		if ce.sense == LESS_OR_EQUAL:
			smodel += ce, c.name
		elif ce.sense == EQUAL:
			c1 = ce
			c1.sense = LESS_OR_EQUAL
			smodel += c1, '{}(L)'.format(c.name)

			c1 = -1.0*c1
			c1.sense = LESS_OR_EQUAL
			smodel += c1, '{}(G)'.format(c.name)
		elif ce.sense == GREATER_OR_EQUAL:
			lee = -1.0*ce
			lee.sense = LESS_OR_EQUAL
			smodel += lee, c.name

	return smodel

def separate_cuts( origMip : Model, stdMip : Model ) -> List[LinExpr]:
	# set of variables active in LP relaxation
	V = [var for var in origMip.vars if abs(var.x)>=1e-4]

	# creating model to separate cuts
	cgsep = Model( solver_name="gurobi", sense=MAXIMIZE )

	U = [ cgsep.add_var(name='u({})'.format(constr.name), lb=0.0, ub=0.99, type=CONTINUOUS)
		for constr in stdMip.constrs ]

	a = [ cgsep.add_var( name='a({})'.format(var.name),
		lb=-maxCutCoef, ub=maxCutCoef, type=INTEGER ) for var in V ]

	f = [ cgsep.add_var( name='f({})'.format(var.name),
		lb=0.0, ub=0.99, type=CONTINUOUS ) for var in V ]

	a0 = cgsep.add_var( name="a0", lb=-maxCutCoef, ub=maxCutCoef, type=INTEGER  )
	f0 = cgsep.add_var( name="f0", lb=0.0, ub=0.99, type=CONTINUOUS )

	Y = list()

	if maxMult != inf:
		Y = [ cgsep.add_var(name='y({})'.format(constr.name), type=BINARY)
			for constr in stdMip.constrs ]

	# objective function:
	cgsep += \
		xsum( var.x*a[j] for j,var in enumerate(V) ) \
		 -a0 - xsum( 1e-4*u for u in U )

	# linking a with us
	for j,var in enumerate(V):
		if 'x(3,4,0)' in var.name:
			print('here')
		col = stdMip.vars[var.idx].column
		cgsep += a[j] + f[j] == \
			xsum( col.coeffs[j]*U[constr.idx] for \
				j,constr in enumerate(col.constrs) ), 'lnkA({})'.format(var.name)

	# lnk y with u
	if len(Y):
		for cons in stdMip.constrs:
			cgsep += U[cons.idx] <= Y[cons.idx], 'lnkUY({})'.format(cons.name)

		cgsep += xsum(Y[cons.idx] for cons in stdMip.constrs) <= maxMult, 'maxMult'

	# linking a0 and f0 with rhs
	rhs = [-constr.expr.const for constr in stdMip.constrs]
	cgsep += a0 + f0 == \
		xsum( v*U[j] for j,v in enumerate(rhs) if abs(v)>1e-7 )

	# some active a
	cgsep += xsum(aa for aa in a) >= 1, 'apos' 

	cgsep.write('cgsep.lp')
	status = cgsep.optimize( max_seconds=1000 )
	
	if status==OPTIMAL or status==FEASIBLE:
		print('Violated cut(s) found ! Best cut:')
		stdout.write('multipliers: \n\t')
		nprint = 0
		for i,u in enumerate(U):
			if abs(u.x)>1e-6:
				stdout.write('{} {}  '.format(u.x, stdMip.constrs[i].name))
				nprint += 1
				if nprint==5:
					stdout.write('\n\t')
					nprint = 0

		stdout.write('\nCut:\n\t')

		nprint = 0
		rhs = round(a0.x)
		lhsSum = 0.0
		for j,aa in enumerate(a):
			if abs(aa.x)>1e-6:
				v = V[j]
				coef = round(aa.x)
				lhsSum += coef*v.x
				stdout.write('{:+} {} '.format(coef, v.name))
				nprint += 1
				if nprint==7:
					nprint = 0
					stdout.write('\n\t')
		stdout.write('<= {}\n'.format(round(rhs)))
		print('violation {}'.format(lhsSum-rhs))


if len(argv)<2:
    print('usage: \n\tcgsep instanceName')
    exit(1)

# original mip
mip1 = Model( solver_name="gurobi" )
mip1.read( argv[1] )

print('original mip has {} variables and {} constraints'.format(mip1.num_cols, mip1.num_rows))

mip1.relax()

it = 0

mip1.write('omip.lp')

# solve LP relaxation
status = mip1.optimize()
print('status: {}'.format(status))
assert status==OPTIMAL
print('obj relax {}'.format(mip1.get_objective_value()))

mip2 = std_model(mip1)
mip2.write('smip.lp')
print('mip in standard form has {} variables and {} constraints'.format(mip2.num_cols, mip2.num_rows))

cuts = separate_cuts(mip1, mip2)

#vim: ts=4 sw=4 et

