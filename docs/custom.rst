Developing Customized Branch-&-Cut algorithms
=============================================

This chapter discusses some features of Python-MIP that allow the
development of improved Branch-&-Cut algorithms by adding application
specific routines to the generic algorithm included in the solvers.

Cut generators
~~~~~~~~~~~~~~

In many applications there are strong formulations that require an
exponential number of constraints. These formulations cannot be direct
handled by the MIP Solver: entering all these constraints at once is
usually not practical. Using cut generators you can interface with the MIP
solver so that at each node of the search tree you can insert only the
violated inequalities, called *cuts*. The problem of discovering these
violated inequalities is called the *Separation Problem*. As an example,
consider the Traveling Salesman Problem. The  compact formulation
(:numref:`tsp-label`) is a rather *weak* formulation. Dual bounds produced
at the root node of the search tree are distant from the optimal solution
cost and improving these bounds requires a potentially intractable number
of branchings. In this case, the culprit are the subtour elimination
constraints involving variables :math:`x` and :math:`y`. A much stronger
formulation can be obtained using the following subtour elimination
constraints:

.. math::

 \sum_{i\in S}\sum_{j \in I\setminus S} x_{i,j} \geq 1 \,\,\, \forall
 S \subset I
