.. _chapQuick:

Quick start
===========

This chapter presents the main components needed to build and optimize models using Python-MIP. A full description of the methods and their parameters can be found at :ref:`Chapter 4 <chapClasses>`.

The first step to enable Python-MIP in your Python code is to add:

.. code-block:: python

   from mip import *

When loaded, Python-MIP will display its installed version: ::

   Using Python-MIP package version 1.6.2

Creating Models
---------------

The model class represents the optimization model.
The code below creates an empty Mixed-Integer Linear Programming problem with default settings.

.. code-block:: python

   m = Model()

By default, the optimization sense is set to *Minimize* and the selected solver is set to CBC. If Gurobi is installed and configured, it will be used instead. You can change the model objective sense or force the selection of a specific solver engine using additional parameters for the constructor:

.. code-block:: python

   m = Model(sense=MAXIMIZE, solver_name=CBC) # use GRB for Gurobi

After creating the model, you should include your decision variables, objective function and constraints. These tasks will be discussed in the next sections.

Variables
~~~~~~~~~

Decision variables are added to the model using the :py:meth:`~mip.Model.add_var` method. Without parameters, a single variable with domain in :math:`\mathbb{R}^+` is created and its reference is returned:

.. code-block:: python

    x = m.add_var()

By using Python list initialization syntax, you can easily create a vector of variables. Let's say that your model will have `n` binary decision variables (n=10 in the example below) indicating if each one of 10 items is selected or not. The code below creates 10 binary variables :code:`y[0]`, ..., :code:`y[n-1]` and stores their references in a list.

.. code-block:: python

    n = 10
    y = [ m.add_var(var_type=BINARY) for i in range(n) ]

Additional variable types are :code:`CONTINUOUS` (default) and :code:`INTEGER`.
Some additional properties that can be specified for variables are their lower and upper bounds (:py:attr:`~mip.Var.lb` and :py:attr:`~mip.Var.ub`, respectively), and names (property :attr:`~mip.Var.name`).
Naming a variable is optional and it is particularly useful if you plan to save you model (see :ref:`save-label`) in .LP or .MPS file formats, for instance.
The following code creates an integer variable named :code:`zCost` which is restricted to be in range :math:`\{-10,\ldots,10\}`.
Note that the variable's reference is stored in a Python variable named :code:`z`.

.. code-block:: python

    z = m.add_var(name='zCost', var_type=INTEGER, lb=-10, ub=10)

You don't need to store references for variables, even though it is usually easier to do so to write constraints.
If you do not store these references, you can get them afterwards using the Model function :func:`~mip.Model.var_by_name`.
The following code retrieves the reference of a variable named :code:`zCost` and sets its upper bound to 5:

.. code-block:: python

   vz = m.var_by_name('zCost')
   vz.ub = 5

Constraints
~~~~~~~~~~~

Constraints are linear expressions involving variables, a sense of :code:`==`, :code:`<=` or :code:`>=` for equal, less or equal and greater or equal, respectively, and a constant.
The constraint :math:`x+y \leq 10` can be easily included within model :code:`m`:

.. code-block:: python

    m += x + y <= 10

Summation expressions can be implemented with the function :py:func:`~mip.xsum`.
If for a knapsack problem with :math:`n` items, each one with weight :math:`w_i`, we would like to include a constraint to select items with binary variables :math:`x_i` respecting the knapsack capacity :math:`c`, then the following code could be used to include this constraint within the model :code:`m`:

.. code-block:: python

    m += xsum(w[i]*x[i] for i in range(n)) <= c

Conditional inclusion of variables in the summation is also easy.
Let's say that only even indexed items are subjected to the capacity constraint:

.. code-block:: python

    m += xsum(w[i]*x[i] for i in range(n) if i%2 == 0) <= c

Finally, it may be useful to name constraints.
To do so is straightforward: include the constraint's name after the linear expression, separating it with a comma.
An example is given below:

.. code-block:: python

    m += xsum(w[i]*x[i] for i in range(n) if i%2 == 0) <= c, 'even_sum'

As with variables, reference of constraints can be retrieved by their names.
Model function :func:`~mip.Model.constr_by_name` is responsible for this:

.. code-block:: python

   constraint = m.constr_by_name('even_sum')

Objective Function
~~~~~~~~~~~~~~~~~~

By default a model is created with the *Minimize* sense.
The following code alters the objective function to :math:`\sum_{i=0}^{n-1} c_ix_i` by setting the :attr:`~mip.Model.objective` attribute of our example model :code:`m`:

.. code-block:: python

   m.objective = xsum(c[i]*x[i] for i in range(n))

To specify whether the goal is to *Minimize* or *Maximize* the objective function, two useful functions were included: :func:`~mip.minimize` and :func:`~mip.maximize`. Below are two usage examples:

.. code-block:: python

   m.objective = minimize(xsum(c[i]*x[i] for i in range(n)))

.. code-block:: python

   m.objective = maximize(xsum(c[i]*x[i] for i in range(n)))

You can also change the optimization direction by setting the :attr:`~mip.Model.sense` model property to :code:`MINIMIZE` or :code:`MAXIMIZE`.

.. _save-label:

Saving, Loading and Checking Model Properties
---------------------------------------------

Model methods :func:`~mip.Model.write` and :func:`~mip.Model.read` can be used to save and load, respectively,
MIP models.
Supported file formats for models are the `LP file format
<https://www.ibm.com/support/knowledgecenter/SSSA5P_12.9.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/tutorials/InteractiveOptimizer/usingLPformat.html>`_, which is more readable and suitable for debugging, and the `MPS file format <https://en.wikipedia.org/wiki/MPS_(format)>`_, which is recommended for extended compatibility, since it is an older and more widely adopted format.
When calling the :meth:`~mip.Model.write` method, the file name extension (.lp or .mps) is used to define the file format.
Therefore, to save a model :code:`m` using the lp file format to the file model.lp we can use:

.. code-block:: python

    m.write('model.lp')

Likewise, we can read a model, which results in creating variables and constraints from the LP or MPS file read.
Once a model is read, all its attributes become available, like the number of variables, constraints and non-zeros in the constraint matrix:

.. code-block:: python

   m.read('model.lp')
   print('model has {} vars, {} constraints and {} nzs'.format(m.num_cols, m.num_rows, m.num_nz))

Optimizing and Querying Optimization Results
--------------------------------------------

MIP solvers execute a Branch-&-Cut (BC) algorithm that in *finite time* will provide the optimal solution.
This time may be, in many cases, too large for your needs.
Fortunately, even when the complete tree search is too expensive, results are often available in the beginning of the search.
Sometimes a feasible solution is produced when the first tree nodes are processed and a lot of additional effort is spent improving the *dual bound*, which is a valid estimate for the cost of the optimal solution.
When this estimate, the lower bound for minimization, matches exactly the cost of the best solution found, the upper bound, the search is concluded.
For practical applications, usually a truncated search is executed.
The :func:`~mip.Model.optimize` method, that executes the optimization of a formulation, accepts optionally processing limits as parameters.
The following code executes the branch-&-cut algorithm to solve a model :code:`m` for up to 300 seconds.

.. code-block:: python
   :linenos:

   m.max_gap = 0.05
   status = m.optimize(max_seconds=300)
   if status == OptimizationStatus.OPTIMAL:
       print('optimal solution cost {} found'.format(m.objective_value))
   elif status == OptimizationStatus.FEASIBLE:
       print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
   elif status == OptimizationStatus.NO_SOLUTION_FOUND:
       print('no feasible solution found, lower bound is: {}'.format(m.objective_bound))
   if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
       print('solution:')
       for v in m.vars:
          if abs(v.x) > 1e-6: # only printing non-zeros
             print('{} : {}'.format(v.name, v.x))

Additional processing limits may be used: :code:`max_nodes` restricts the maximum number of explored nodes in the search tree and :code:`max_solutions` finishes the BC algorithm after a number of feasible solutions are obtained.
It is also wise to specify how tight the bounds should be to conclude the search.
The model attribute :attr:`~mip.Model.max_mip_gap` specifies the allowable percentage deviation of the upper bound from the lower bound for concluding the search.
In our example, whenever the distance of the lower and upper bounds is less or equal 5\% (see line 1), the search can be finished.

The :meth:`~mip.Model.optimize` method returns the status
(:class:`~mip.OptimizationStatus`) of the BC search:
:attr:`~mip.OptimizationStatus.OPTIMAL` if the search was concluded and the optimal solution was found; :attr:`~mip.OptimizationStatus.FEASIBLE` if a feasible solution was found but there was no
time to prove whether this solution was optimal or not;
:attr:`~mip.OptimizationStatus.NO_SOLUTION_FOUND` if in the truncated search no solution was found; :attr:`~mip.OptimizationStatus.INFEASIBLE` or :attr:`~mip.OptimizationStatus.INT_INFEASIBLE` if no feasible solution exists for the model;
:attr:`~mip.OptimizationStatus.UNBOUNDED` if there are missing constraints or :attr:`~mip.OptimizationStatus.ERROR` if
some error occurred during optimization. In the example above, if a feasible
solution is available (line 8), variables which have value different from zero
are printed. Observe also that even when no feasible solution is available
the dual bound (lower bound in the case of minimization) is available (line 8): if a truncated execution was performed, i.e., the solver stopped due to the time limit, you can check this dual bound which is an estimate of
the quality of the solution found checking the :attr:`~mip.Model.gap`
property.

During the tree search, it is often the case that many different feasible solutions
are found. The solver engine stores this solutions in a solution pool. The following code
prints all routes found while optimizing the :ref:`Traveling Salesman Problem <tsp-label>`.


.. code-block:: python

    for k in range(model.num_solutions):
        print('route {} with length {}'.format(k, model.objective_values[k]))
        for (i, j) in product(range(n), range(n)):
            if x[i][j].xi(k) >= 0.98:
                print('\tarc ({},{})'.format(i,j))



Performance Tuning
~~~~~~~~~~~~~~~~~~

Tree search algorithms of MIP solvers deliver a set of improved feasible
solutions and lower bounds. Depending on your application you will
be more interested in the quick production of feasible solutions than in improved
lower bounds that may require expensive computations, even if in the long term
these computations prove worthy to prove the optimality of the solution found.
The model property  :attr:`~mip.Model.emphasis` provides three different settings:

0. default setting:
    tries to balance between the search of improved feasible
    solutions and improved lower bounds;

1. feasibility:
    focus on finding improved feasible solutions in the
    first moments of the search process, activates heuristics;

2. optimality:
    activates procedures that produce improved lower bounds, focusing
    in pruning the search tree even if the production of the first feasible solutions
    is delayed.

Changing this setting to 1 or 2 triggers the activation/deactivation of
several algorithms that are processed at each node of the search tree that
impact the solver performance. Even though in average these settings
change the solver performance as described previously, depending on your
formulation the impact of these changes may be very different and it is
usually worth to check the solver behavior with these different settings
in your application.

Another parameter that may be worth tuning is the :attr:`~mip.Model.cuts`
attribute, that controls how much computational effort should be spent in generating
cutting planes.


