#!/usr/bin/python
# -*- coding: utf-8 -*-

import pdb
from mip.model import *

EPS = 10e-4
SOLVER = GUROBI


def cg():
    """
    Simple column generation implementation for a Cutting Stock Problem
    """

    L = 250  # bar length
    m = 4  # number of requests
    w = [187, 119, 74, 90]  # size of each item
    b = [1, 2, 2, 1]  # demand for each item

    # creating models and auxiliary lists
    master = Model(SOLVER)
    lambdas = []
    constraints = []

    # creating an initial pattern (which cut one item per bar)
    # to provide the restricted master problem with a feasible solution
    for i in range(m):
        lambdas.append(master.add_var(obj=1, name='lambda_%d' % (len(lambdas) + 1)))

    # creating constraints
    for i in range(m):
        constraints.append(master.add_constr(lambdas[i] >= b[i], name='i_%d' % (i + 1)))

    # creating the pricing problem
    pricing = Model(SOLVER)

    # creating pricing variables
    a = []
    for i in range(m):
        a.append(pricing.add_var(obj=0, type=INTEGER, name='a_%d' % (i + 1)))

    # creating pricing constraint
    pricing.add_constr(xsum(w[i] * a[i] for i in range(m)) <= L, name='bar_length')

    pricing.write('pricing.lp')

    new_vars = True
    while (new_vars):

        ##########
        # STEP 1: solving restricted master problem
        ##########

        master.optimize()
        master.write('master.lp')

        # printing dual values
        print_solution(master)
        print('pi = ', end='')
        print([constraints[i].pi for i in range(m)])
        print('')

        ##########
        # STEP 2: updating pricing objective with dual values from master
        ##########

        pricing.set_objective(1)
        for i in range(m):
            a[i].obj = -constraints[i].pi

        # solving pricing problem
        pricing.optimize()

        # printing pricing solution
        z_val = pricing.get_objective()
        print('Pricing:')
        print('    z =  {z_val}'.format(**locals()))
        print('    a = ', end='')
        print([v.x for v in pricing.vars])
        print('')

        ##########
        # STEP 3: adding the new columns
        ##########

        # checking if columns with negative reduced cost were produced and
        # adding them into the restricted master problem
        if pricing.get_objective_value() < - EPS:
            coeffs = [a[i].x for i in range(m)]
            column = Column(constraints, coeffs)
            lambdas.append(master.add_var(obj=1, column=column, name='lambda_%d' % (len(lambdas) + 1)))

            print('new pattern = {coeffs}'.format(**locals()))

        # if no column with negative reduced cost was produced, then linear
        # relaxation of the restricted master problem is solved
        else:
            new_vars = False

        pricing.write('pricing.lp')
        # pdb.set_trace()

    print_solution(master)


def kantorovich():
    """
    Simple implementation of the compact formulation from Kantorovich for the problem
    """

    N = 10  # maximum number of bars
    L = 250  # bar length
    m = 4  # number of requests
    w = [187, 119, 74, 90]  # size of each item
    b = [1, 2, 2, 1]  # demand for each item

    # creating the model (note that the linear relaxation is solved)
    model = Model(SOLVER)
    x = {(i, j): model.add_var(obj=0, type=CONTINUOUS, name="x[%d,%d]" % (i, j)) for i in range(m) for j in range(N)}
    y = {j: model.add_var(obj=1, type=CONTINUOUS, name="y[%d]" % j) for j in range(N)}

    # constraints
    for i in range(m):
        model.add_constr(xsum(x[i, j] for j in range(N)) >= b[i])
    for j in range(N):
        model.add_constr(xsum(w[i] * x[i, j] for i in range(m)) <= L * y[j])

    # additional constraint to reduce symmetry
    for j in range(1, N):
        model.add_constr(y[j - 1] >= y[j])

    # optimizing the model and printing solution
    model.optimize()
    print_solution(model)


def print_solution(model):
    objective = model.get_objective()
    print('')
    print('Objective:\n    {objective:.3}'.format(**locals()))
    print('Solution:')
    for v in model.vars:
        if v.x > EPS:
            print('    {v.name} = {v.x:.3}'.format(**locals()))
    print('')


if __name__ == "__main__":
    cg()
    # kantorovich()
