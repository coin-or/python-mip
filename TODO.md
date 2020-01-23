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
Input and output of problem models using numpy would allow an easier integration with numba, for instance, which would be great for maximum performance.

Column generation
-----------------

Add example(s) with column generation

Quadratic programming
---------------------

It would be easy to add the possibility of modelling x1*x2 if both are binary variables since they can be easily internally replaced by an auxiliary variable y that represents the linearization of this expression. 

Jupyter Notebooks
-----------------

Add some Jupyter notebooks with visualization of solutions (routes). Jupyter notebooks also allow a better output of tables for solution visualization. It seems that nbsphinx and jupyter-sphinx are good tools.

Method to generate cutting planes
---------------------------------

It would be nice to be able to call the CBC procedures for generating specific types of cuts. This would ease the comparison of new cuts with the existing ones and also ease the development of cutting plane algorithms where one wants to evaluate the performance of a new type of cut when included in an existing cutting plane framework. One drawback is that it would be a CBC only feature since gurobi does not allows access to the generated cuts.

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


