from mip.model import *
from sys import argv
from typing import List

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

def separate_cuts( mip : Model ) -> List[LinExpr]:
	# set of variables active in LP relaxation
	V = [var for var in omip.vars if var.x>=1e-4]

	# creating model to separate cuts
	cgsep = Model( solver_name="gurobi", sense=MAXIMIZE )

	U = [ cgsep.add_var(name='u({})'.format(constr.name), lb=0.0, ub=0.5, type=CONTINUOUS)
		for constr in mip.constrs ]

	a = [ cgsep.add_var( name='a({})'.format(var.name),
		lb=-10, ub=10, type=INTEGER ) for var in V ]

	f = [ cgsep.add_var( name='f({})'.format(var.name),
		lb=0.0, ub=0.99, type=CONTINUOUS ) for var in V ]

	a0 = cgsep.add_var( name="a0", lb=-10, ub=10, type=INTEGER  )
	f0 = cgsep.add_var( name="f0", lb=0.0, ub=0.99, type=CONTINUOUS )

	# objective function:
	cgsep += \
		xsum( var.x*a[j] for j,var in enumerate(V) ) \
		 -a0 - xsum( 1e-4*u for u in U )

	# linking a with us
	for j,var in enumerate(V):
		col = var.column
		cgsep += a[j] + f[j] == \
			xsum( col.coeffs[j]*U[constr.idx] for \
				j,constr in enumerate(col.constrs) ), 'lnkA({})'.format(var.name)
	
	# linking a0 and f0 with rhs
	rhs = [-constr.expr.const for constr in mip.constrs]
	cgsep += a0 + f0 == \
		xsum( v*U[j] for j,v in enumerate(rhs) if abs(v)>1e-7 )
	
	cgsep.write('cgsep.lp')

if len(argv)<2:
    print('usage: \n\tcgsep instanceName')
    exit(1)

# original mip
omip = Model( solver_name="gurobi" )
omip.read( argv[1] )

print('original mip has {} variables and {} constraints'.format(omip.num_cols, omip.num_rows))

omip.relax()

it = 0

omip.write('omip.lp')

# solve LP relaxation
status = omip.optimize()
assert status==OPTIMAL
print('obj relax {}'.format(omip.get_objective_value()))

smip = std_model(omip)
print('mip in standard form has {} variables and {} constraints'.format(smip.num_cols, smip.num_rows))

cuts = separate_cuts(smip)

#vim: ts=4 sw=4 et

