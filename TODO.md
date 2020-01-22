---
title: Commented TODO list for Python-MIP
---


Stopping criterion 
------------------

Iterations/time without improving feasible solution
similar to #40

Numpy and Scipy support
-----------------------

Improvement #12

Some problems are more easily specified in matrix notation.
Input and output of problem data using numpy would allow an easier integration with numba, for instance.

Column generation
-----------------

Add example(s) with column generation

Rapsberry pi
------------

Support this platform - right now no developer access to this hardware

More CBC tests (in CBC)
-----------------------

The stability of Python-MIP depends directly on the stability of the COIN-OR CBC solver. Thus, it would be good to improve the automated tests in CBC, including modern CBC features such as lazy constraints in additional tests.


- AVX optional in CBC
- feasopt
- cg examples
- option to strengthen formulation with cuts
- cplex support


