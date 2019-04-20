Quick start
===========

This chapter presents the main components needed to build and optimize
models in Python-MIP. A full description of the methods and their
parameters can be found at :ref:`Chapter 4 <chapClasses>`.

The first step to enable Python-MIP in your Python code is to add:

.. code-block:: python

   from mip.model import *

when loaded, Python-MIP will display its installed version: ::

   Using Python-MIP package version 1.0.29

Model 
-----

The model class represents the optimization model. The code:

.. code-block:: python

   m = Model()

Creates an empty Mixed Integer Linear Programming problem settings some
defaults: the optimization sense is set to *Minimize* and the selected
solver is set to CBC or to Gurobi, if this is installed and configured.
You can change the model objective sense or force a solver to be selected
at the model creation using additional parameters for the constructor:

.. code-block:: python

   m = Model(sense=MAXIMIZE, solver_name="cbc")

After creating the model, you should include your decision variables, objective
function and constraints. These tasks will be discussed in the next sections:

Variables 
---------

Decision variables are added to the model using the :code:`add_var`
method. Without parameters, a single variable with domain in 
:math:`\mathbb{R}^+` is created and its reference is returned:

.. code-block:: python

    x = m.add_var()

Using Python list initialization syntax you and easily create a vector of
variables. Let's say that your model will have `n` binary decision
variables indicating if specified items were selected or not:

.. code-block:: python 

    y = [m.add_var(var_type=BINARY) for i in range(n)]

This code would create binary variables :code:`y[0]`, ..., :code:`y[n-1]`.
Additional variable types are :code:`CONTINUOUS` (default) and :code:`INTEGER`.
Some additional properties that can be specified for variables are their lower
and upper bounds (:code:`lb` and :code:`ub`, respectively) and their names.
Entering variable names is optional but specially useful if you plan to save you
model in .LP or .MPS file formats, for instance.  The following code creates a
variable named :code:`zCost` that is restricted to be integral in the range
:math:`\{-10,\ldots,10\}`, the reference for this variable is stored in a
Python variable named :code:`z`.

.. code-block:: python

    z=m.add_var(name='zCost', var_type=INTEGER, lb=-10, ub=10)

You don't need to store references for variables, even though it is usually
easier to do so to write the constraints. If you do not store these references,
you can get them after using the :code:`get_var_by_name` Model method. The
following code retrieves a reference to a variable named :code:`zCost` and sets
the upper bound for this variable to 5:

.. code-block:: python

   vz = m.get_var_by_name('zCost') 
   vz.ub = 5

Constraints
-----------

Constraints are linear expressions involving variables, a sense, ==, <= or >= for 
equal, less or equal and greater or equal, respectively  and a constant 
in the right-hand side. The addition of constraint :math:`x+y \leq 10` to model
:code:`m` can be done with:

.. code-block:: python 

    m += x + y <= 10

Summation expressions can be used with the function :code:`xsum`. If for a knapsack problem
with :math:`n` items, each one with weight :math:`w_i` we would like to select items with 
binary variables :math:`x_i` respecting the knapsack capacity :math:`c`, then the following 
code could be used to enter this constraint to our model :code:`m`:

.. code-block:: python 

    m += xsum(w[i]*x[i] for i in range(n)) <= c

Conditional inclusion of variables in the summation is also easy. Let's say that only 
even indexed items are subjected to the capacity constraint:

.. code-block:: python 

    mip += xsum(w[i]*x[i] for i in range(n) if i%2==0) <= c

Objective Function
------------------

By default a model is created with the *Minimize* sense. You can change by
setting the :code:`sense` model property to :code:`MAXIMIZE`, or just
multiply the objective function by -1. The following code adds :math:`n`
:math:`x` variables to the objective function, each one with cost
:math:`c_i`:

.. code-block:: python

   m += xsum(c[i]*x[i] for i in range(n))


