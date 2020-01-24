#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Simple column generation implementation for a Cutting Stock Problem
"""

from mip import Model, xsum, Column, CONTINUOUS, INTEGER

L = 250  # bar length
m = 4  # number of requests
w = [187, 119, 74, 90]  # size of each item
b = [1, 2, 2, 1]  # demand for each item

# creating master model
master = Model()

# creating an initial set of patterns which cut one item per bar
# to provide the restricted master problem with a feasible solution
lambdas = [master.add_var(obj=1, name='lambda_%d' % (j + 1))
           for j in range(m)]

# creating constraints
constraints = []
for i in range(m):
    constraints.append(master.add_constr(lambdas[i] >= b[i], name='i_%d' % (i + 1)))

# creating the pricing problem
pricing = Model()

# creating pricing variables
a = [pricing.add_var(obj=0, var_type=INTEGER, name='a_%d' % (i + 1)) for i in range(m)]

# creating pricing constraint
pricing += xsum(w[i] * a[i] for i in range(m)) <= L, 'bar_length'

new_vars = True
while new_vars:

    ##########
    # STEP 1: solving restricted master problem
    ##########

    master.optimize()

    ##########
    # STEP 2: updating pricing objective with dual values from master
    ##########

    pricing += 1 - xsum(constraints[i].pi * a[i] for i in range(m))

    # solving pricing problem
    pricing.optimize()

    # printing pricing solution
    z_val = pricing.objective_value
    print('Pricing solution:')
    print('    z =  {z_val}'.format(**locals()))
    print('    a = ', end='')
    print([v.x for v in pricing.vars])
    print('')

    ##########
    # STEP 3: adding the new columns (if any is obtained with negative reduced cost)
    ##########

    # checking if columns with negative reduced cost were produced and
    # adding them into the restricted master problem
    if pricing.objective_value < - 1e-5:
        pattern = [a[i].x for i in range(m)]
        column = Column(constraints, pattern)
        lambdas.append(master.add_var(obj=1, column=column,
                                      name='lambda_%d' % (len(lambdas) + 1)))

        print('new pattern = {pattern}'.format(**locals()))

    # if no column with negative reduced cost was produced, then linear
    # relaxation of the restricted master problem is solved
    else:
        new_vars = False

    pricing.write('pricing.lp')

# printing the solution
print('')
print('Objective value: {master.objective_value:.3}'.format(**locals()))
print('Solution: ', end='')
for v in lambdas:
    if v.x > 1e-6:
        print('{v.name} = {v.x:.3}  {v.column}'.format(**locals()))
        print('          ', end='')

# sanity checks
master.check_optimization_results()
