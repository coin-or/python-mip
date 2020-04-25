Classes
=======

.. _chapClasses:

.. automodule:: mip.callbacks
.. automodule:: mip.constants
.. automodule:: mip.exceptions
.. automodule:: mip.model

Model
-----

.. autoclass:: mip.model.Model
    :members:

LinExpr
-------
.. autoclass:: mip.model.LinExpr
    :members:

Var
---
.. autoclass:: mip.model.Var
    :members:

Constr
------
.. autoclass:: mip.model.Constr
    :members:

Column
------
.. autoclass:: mip.model.Column
    :members:

VarList
-------
.. autoclass:: mip.model.VarList
    :members:

ConstrList
----------
.. autoclass:: mip.model.ConstrList
    :members:

ConstrsGenerator
----------------
.. autoclass:: mip.callbacks.ConstrsGenerator
    :members:

IncumbentUpdater
----------------
.. autoclass:: mip.callbacks.IncumbentUpdater
    :members:

CutType
--------
.. autoclass:: mip.constants.CutType
    :members:

CutPool
-------
.. autoclass:: mip.callbacks.CutPool
    :members:

OptimizationStatus
------------------
.. autoclass:: mip.constants.OptimizationStatus
    :members:

LP_Method
---------
.. autoclass:: mip.constants.LP_Method
   :members:



ProgressLog
-----------
.. autoclass:: mip.model.ProgressLog
    :members:

Exceptions
-----------

.. autoclass:: mip.exceptions.MipBaseException
.. autoclass:: mip.exceptions.ProgrammingError
.. autoclass:: mip.exceptions.InterfacingError
.. autoclass:: mip.exceptions.InvalidLinExpr
.. autoclass:: mip.exceptions.InvalidParameter
.. autoclass:: mip.exceptions.ParameterNotAvailable
.. autoclass:: mip.exceptions.InfeasibleSolution
.. autoclass:: mip.exceptions.SolutionNotAvailable

Useful functions
----------------

.. automethod:: mip.model.minimize
.. automethod:: mip.model.maximize
.. automethod:: mip.model.xsum
