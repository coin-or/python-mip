.. _chapExamples:

Modeling Examples
=================

This chapter includes commented examples on modeling and solving optimization
problems with Python-MIP.

The 0/1 Knapsack Problem
------------------------
 
As a first example, consider the solution of the 0/1 knapsack problem:
given a set :math:`I` of items, each one with a weight :math:`w_i`  and
estimated profit :math:`p_i`, one wants to select a subset with maximum
profit such that the summation of the weights of the selected items is
less or equal to the knapsack capacity :math:`c`.
Considering a set of decision binary variables :math:`x_i` that receive
value 1 if the :math:`i`-th item is selected, or 0 if not, the resulting
mathematical programming formulation is: 

.. math::
   
    \textrm{Maximize: }   &  \\
                                   &  \sum_{i \in I} p_i \cdot x_i  \\
    \textrm{Subject to: } & \\
                                   &  \sum_{i \in I} w_i \cdot x_i \leq c  \\
                                   &  x_i \in \{0,1\} \,\,\, \forall i \in I

The following python code creates, optimizes and prints the optimal solution for the
0/1 knapsack problem

.. code-block:: python
    :linenos:

    from mip.model import * 
    p = [10, 13, 18, 31,  7, 15] 
    w = [11, 15, 20, 35, 10, 33] 
    c = 40 
    n = len(w) 
    m = Model('knapsack', MAXIMIZE)
    x = [m.add_var(var_type='B') for i in range(n)] 
    m += xsum(p[i]*x[i] for i in range(n) ) 
    m += xsum(w[i]*x[i] for i in range(n) ) <= c
    m.optimize() 
    selected=[i for i in range(n) if x[i].x>=0.99]
    print('selected items: {}'.format(selected))

Lines 1-5 load problem data. In line 6 an empty maximization
model m with the (optional) name of "knapsack" is created. Line 7 adds the
binary decision variables to model m. Line 8 defines the objective
function of this model and Line 9 adds the capacity constraint. The model
is optimized in line 10 and the solution, a list of the selected items, is
computed at line 11.

.. _tsp-label:

The Traveling Salesman Problem
------------------------------

The traveling salesman problem (TSP) is one of the most studied combinatorial
optimization problems. To to illustrate this problem, consider that you
will spend some time in Belgium and wish to visit some of its main tourist
attractions, depicted in the map bellow:

.. image:: images/belgium-tourism-14.png
    :width: 60%
    :align: center

You want to find the shortest possible tour to visit all these places. More
formally, considering  :math:`n` points :math:`I=\{0,\ldots,n-1\}` and
a distance matrix :math:`D_{n \times n}` with elements :math:`d_{i,j} \in
\mathbb{R}^+`, a solution consists in a set of exactly :math:`n` (origin, 
destination) pairs indicating the itinerary of your trip, resulting in
the following formulation:

.. math::

    \textrm{Minimize: }   &  \\ 
    &  \sum_{i \in I, j \in I : i \neq j} d_{i,j} \ldotp x_{i,j} \\
    \textrm{Subject to: }   &  \\ 
    & \sum_{j \in I : i \neq j} x_{i,j} = 1 \,\,\, \forall i \in I  \\
    & \sum_{i \in I : i \neq j} x_{i,j} = 1 \,\,\, \forall j \in I \\
    & y_{i} -(n+1)\ldotp x_{i,j} \geq y_{j} -n  \,\,\, \forall i \in I\setminus \{0\}, j \in I\setminus \{0,i\}\\
    & x_{i,j} \in \{0,1\} \,\,\, \forall i \in J, j \in I\setminus \{j\} \\
    & y_i \geq 0 \,\,\, \forall i \in I

The first two sets of constraints enforce that we leave and arrive only
once at each point. The optimal solution for the problem including only
these constraints could result in a solution with sub-tours, such as the
one bellow.

.. image:: images/belgium-tourism-14-subtour.png 
    :width: 60%
    :align: center

To enforce the production of connected routes, additional variables
:math:`y_{i} \geq 0` are included in the model indicating the
sequential order of each point in the produced route. Point zero is
arbitrarily selected as the initial point and conditional constraints
linking variables :math:`x_{i,j},y_{i}` and :math:`y_{j}` ensure that the
selection of the arc :math:`x_{i,j}` implies that :math:`y_{j}\geq y_{i}+1`.

The Python code to create, optimize and print the optimal route for the TSP is
included bellow:


.. code-block:: python
    :linenos:

    from tspdata import TSPData
    from sys import argv
    from mip.model import *
    from mip.constants import *
    inst = TSPData(argv[1])
    n = inst.n
    d = inst.d
    model = Model()
    x = [ [ model.add_var(var_type=BINARY) for j in range(n) ] for i in range(n) ]
    y = [ model.add_var() for i in range(n) ]
    model += xsum( d[i][j]*x[i][j] for j in range(n) for i in range(n) )
    for i in range(n):
        model += xsum( x[j][i] for j in range(n) if j != i ) == 1
    for i in range(n):
        model += xsum( x[i][j] for j in range(n) if j != i ) == 1
    for i in range(1, n):
        for j in [x for x in range(1, n) if x!=i]:
            model += y[i]  - (n+1)*x[i][j] >=  y[j] -n
    model.optimize(max_seconds=30)
    arcs = [(i,j) for i in range(n) for j in range(n) if x[i][j].x >= 0.99]
    print('optimal route : {}'.format(arcs))

This `example <https://raw.githubusercontent.com/coin-or/python-mip/master/examples/tsp-compact.py>`_ is included in the Python-MIP package in the example folder
Additional code to load the problem data (called from line 5) is included in `tspdata.py <https://raw.githubusercontent.com/coin-or/python-mip/master/examples/tspdata.py>`_. 
File `belgium-tourism-14.tsp <https://raw.githubusercontent.com/coin-or/python-mip/master/examples/belgium-tourism-14.tsp>`_ contains the coordinates
of the cities included in the example. To produce the optimal tourist tour for our Belgium example just enter:

.. code-block:: bash

    python tsp-compact.py belgium-tourism-14.tsp

In the command line. Follows an explanation of the tsp-compact code: line
10 creates the main binary decision variables for the selection of arcs
and line 11 creates the auxiliary continuous variables. Differently
from the :math:`x` variables, :math:`y` variables are not required to be
binary or integral, they can be declared just as continuous variables, the
default variable type. In this case, the parameter :code:`var_type` can be
omitted from the :code:`add_var` call. Line 11 sets the total traveled
distance as objective function and lines 12-18 include the constraints. In
line 19 we call the optimizer specifying a time limit of 30 seconds. This
will surely not be necessary for our Belgium example, which will be solved
instantly, but may be important for larger problems: even though high
quality solutions may be found very quickly by the MIP solver, the time
required to *prove* that the current solution is optimal may be very
large. With a time limit, the search is truncated and the best solution
found during the search is reported. Finally, the optimal solution for our
trip has length 547 and is depicted bellow:

.. image:: ./images/belgium-tourism-14-opt-547.png
    :width: 60%
    :align: center

