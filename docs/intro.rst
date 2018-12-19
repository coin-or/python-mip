Introduction
============

The MIP python package provides a solver independent interface to Mixed-Integer
Linear Programming solvers. The basic functionality allows the modelling,
optimization, problem modification, re-optimization and solution query.

As a first example, consider the solution of the 0/1 knapsack problem: given a
set :math:`I` each one with a weight :math:`w_i`  and estimated profit
:math:`p_i`, one wants to select a subset with maximum profit such that the
summation of the weights of the selected items is less or equal to the capacity
:math:`c` of the knapsack. The resulting mathematical programming formulation
is: 

.. math::
   
    \textrm{\textit{Maximize}: }   &  \\
                                   &  \sum_{i \in I} p_i \cdot x_i  \\
    \textrm{\textit{Subject to}: } & \\
                                   &  \sum_{i \in I} w_i \cdot x_i \leq c  

The following python code creates, optimizes and prints the optimal solution for the
0/1 knapsack problem::
    
    from mip.model import *
    # input data
    p = [10, 13, 18, 31,  7, 15]
    w = [11, 15, 20, 35, 10, 33]
    c = 40
    n = len(v)
    # creates the model
    m = Model('knapsack', MAXIMIZE)
    # adds binary variables
    x = [m.add_var(type=BINARY) for i in range(n)]
    # adding the objective function
    m += xsum(p[i]*x[i] for i in range(n) )
    # capacity constraint
    m += xsum(w[i]*x[i] for i in range(n) ) <= c
    # optimizes and prints the solution
    m.optimize()
    selected=[i for i in range(n) if x[i].x>=0.99]
    print('selected items: {}'.format(selected))

