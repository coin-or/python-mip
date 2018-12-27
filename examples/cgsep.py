from mip.model import *
from sys import argv
from typing import List

def std_model( omip : Model ) -> Model:
	smodel = Model( name=omip.name, sense=omip.sense, solver_name=omip.solver_name )
	for v in omip.vars:
		smodel.add_var( v.name, v.lb, v.ub, v.obj, v.type )

	for c in omip.constrs:
		ce = c.expr
		if ce.sense == LESS_OR_EQUAL:
			smodel += ce, c.name
		elif ce.sense == EQUAL:
			c1 = ce
			c1.sense = LESS_OR_EQUAL
			smodel += c1, '{}(L)'.format(c.name)
			c1.sense = GREATER_OR_EQUAL
			smodel += c1, '{}(G)'.format(c.name)
		elif ce.sense == GREATER_OR_EQUAL:
			smodel += -1.0*ce, c.name

	return smodel

def separateCuts(omip : Model) -> List[LinExpr]:
	# set of variables active in LP relaxation
	V = [var for var in omip.vars if var.x>=1e-4]

	# creating model to separate cuts
	cgsep = Model( solver_name="cbc", sense=MAXIMIZE )

	varsConstr = list()
	for constr in omip.constrs:
		varsConstr.append( list() )

	for constr in omip.constrs:
		expr = constr.expr
		if expr.sense == EQUAL:
			newVarInfo = ( constr.idx, 
			 cgsep.add_var(name='u({}L)'.format(constr.name), lb=0.0, ub=0.99, type=CONTINUOUS), 1.0 )
			varsConstr[constr.idx].append( newVarInfo )
			newVarInfo = ( constr.idx, 
			 cgsep.add_var(name='u({}G)'.format(constr.name), lb=0.0, ub=0.99, type=CONTINUOUS), -1.0 )
			varsConstr[constr.idx].append( newVarInfo )
		elif expr.sense == LESS_OR_EQUAL:
			newVarInfo = ( constr.idx, 
			 cgsep.add_var(name='u({})'.format(constr.name), lb=0.0, ub=0.99, type=CONTINUOUS), 1.0 )
			varsConstr[constr.idx].append( newVarInfo )
		elif expr.sense == GREATER_OR_EQUAL:
			newVarInfo = ( constr.idx, 
			 cgsep.add_var(name='u({})'.format(constr.name), lb=0.0, ub=0.99, type=CONTINUOUS), -1.0 )
			varsConstr[constr.idx].append( newVarInfo )


	#u = [ cgsep.add_var( , 
#		   ) for constr in omip.constrs ]

	a = [ cgsep.add_var( name='a({})'.format(var.name),
		lb=-10, ub=10, type=INTEGER ) for var in V ]

	f = [ cgsep.add_var( name='f({})'.format(var.name),
		lb=0.0, ub=0.99, type=CONTINUOUS ) for var in V ]

	a0 = cgsep.add_var( name="a0", lb=-10, ub=10 )
	f0 = cgsep.add_var( name="f0", lb=0.0, ub=0.99 )

	# objective function:
	#cgsep += 
	#	xsum( var.x*a[j] for j,var in enumerate(V) ) \
	#	 -a0 + xsum()



if len(argv)<2:
    print('usage: \n\tcgsep instanceName')
    exit(1)

# original mip
omip = Model( solver_name="cbc" )
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



status = smip.optimize()
assert status==OPTIMAL
print('obj relax {}'.format(smip.get_objective_value()))

smip.write('smip.lp')

"""
print('at iteration {} obj value is {}'.format(it, omip.get_objective_value()))
"""

#vim: ts=4 sw=4 et

