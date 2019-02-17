Introduction
============

The Python MIP package provides an interface to Mixed-Integer Linear
Programming solvers. The basic functionality allows the modelling,
optimization, problem modification, re-optimization and solution query.
Advanced features such as cut callbacks, MIPStarts and access to the
solution pool are also available. It comes with the open source COIN-OR
Branch-and-Cut solver CBC but also works with the commercial solver
Gurobi.

As a first example, consider the solution of the 0/1 knapsack problem:
given a set :math:`I` of items, each one with a weight :math:`w_i`  and
estimated profit :math:`p_i`, one wants to select a subset with maximum
profit such that the summation of the weights of the selected items is
less or equal to the knapsack capacity :math:`c`.
Considering a set of decision binary variables :math:`x_i` that receive
value 1 if the :math:`i`-th item is selected, or 0 if not, the resulting
mathematical programming formulation is: 

.. math::
   
    \textrm{\textit{Maximize}: }   &  \\
                                   &  \sum_{i \in I} p_i \cdot x_i  \\
    \textrm{\textit{Subject to}: } & \\
                                   &  \sum_{i \in I} w_i \cdot x_i \leq c  

The following python code creates, optimizes and prints the optimal solution for the
0/1 knapsack problem

.. code-block:: python
    :linenos:

    from mip.model import *
    p = [10, 13, 18, 31,  7, 15]
    w = [11, 15, 20, 35, 10, 33]
    c = 40
    n = len(v)
    m = Model('knapsack', MAXIMIZE)
    x = [m.add_var(type=BINARY) for i in range(n)]
    m += xsum(p[i]*x[i] for i in range(n) )
    m += xsum(w[i]*x[i] for i in range(n) ) <= c
    m.optimize()
    selected=[i for i in range(n) if x[i].x>=0.99]
    print('selected items: {}'.format(selected))

Lines 1-5 are just to load problem data. In line 6 an empty maximization
model m with the (optional) name of "knapsack" is created. Line 7 adds the
binary decision variables to model m. Line 8 defines the objective
function of this model and Line 9 adds the capacity constraint. The model
is optimized in line 10 and the solution, a list of the selected items, is
computed at line 11.

