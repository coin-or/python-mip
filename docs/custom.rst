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
TSP formulation can be written as follows: consider a graph
:math:`G=(N,A)` where :math:`N` is the set of nodes and :math:`A` is the
set of directed edges with associated traveling costs :math:`c_a \in A`.
Selection of arcs is done with binary variables :math:`x_a \,\,\, \forall
a \in A`. Consider also that edges arriving and leaving a node :math:`n`
are indicated in :math:`A^+_n` and :math:`A^-_n`, respectively. The
complete formulation follows:


.. math::

  \textrm{Minimize:} &  \\
   & \sum_{a \in A} c_a\ldotp x_a \\
  \textrm{Subject to:} &  \\
   & \sum_{a \in A^+_n} x_a = 1 \,\,\, \forall n \in N \\
   & \sum_{a \in A^-_n} x_a = 1 \,\,\, \forall n \in N \\
 & \sum_{i\in S}\sum_{j \in I\setminus S} x_{i,j} + \sum_{i\in I\setminus S}\sum_{j \in S} x_{i,j} \geq 2 \,\,\, \forall
 S \subset I \\
     & x_a \in \{0,1\} \,\,\, \forall a \in A

The third type of constraint, named sub-tour elimination constraints, is
stated for *every subset* of nodes. As the number of these constraints is
:math:`O(2^{|N|})`, this formulation is not practical to be directly
handled by the MIP solver. In the *Cutting Plane* method, one can start
solving the formulation with the two first constraint types and insert
only violated sub-tour elimination constraints.

As an example, consider the following graph:

.. image:: tspG.pdf
    :width: 45%
    :align: center

The optimal LP relaxation of the previous formulation without the sub-tour
elimination constraints has cost 237:

.. image:: tspRoot.pdf
    :width: 45%
    :align: center

As it can be seen, there are tree disconnected sub-tours. Two of these
include only two nodes. Forbidding sub-tours of size 2 is quite easy: in
this case we only need to include the additional constraints:
:math:`x_{(d,e)}+x_{(e,d)}\leq 1` and :math:`x_{(c,f)}+x_{(f,c)}\leq 1`.

Optimizing with these two additional constraints out objective value would
increase to 244 and the following new solution would be generated:

.. image:: tspNo2Sub.pdf
    :width: 45%
    :align: center

Now there are sub-tours of size 3 and 4. Let's consider the sub-tour
defined by nodes :math:`S=\{a,b,g\}`. To eliminate this sub-tour we need
to include a constraint stating that elements *in* :math:`S` should have
two arcs linking with elements *outside* :math:`S` (:math:`N\setminus S`), one for
entering this subset and another for leaving.
Arcs connecting :math:`S` to the remaining nodes are show bellow:

.. image:: tspC.pdf
    :width: 45%
    :align: center

Our cut, in this case, would be :math:`x_{(a,d)} + x_{(d,a)} + x_{(d,b)} + x_{(b,d)} + x_{(a,c)} + x_{(c,a)} + x_{(g,e)} + x_{(e,g)} + x_{(g,f)} + x_{(f,g)} + x_{(b,e)} + x_{(e,b)} \geq 2`. 
Adding it to our model increases the objective value to 261 and generates the following solution, now with fractional values
for some variables:

.. image:: tspSt1.pdf
    :width: 45%
    :align: center




