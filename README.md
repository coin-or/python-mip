# Python MIP (Mixed-Integer Linear Programming) Tools

Python MIP is a collection of tools for python the modelling and solution
of Mixed-Integer linear programs. MIP syntax was inspired by Pulp and the
Gurobi Python API. Porting Pulp and Gurobi models should be quite easy.

Some of the main features of MIP are:

* multi solver: works with different MIP solvers. Right now Gurobi and CBC
  are supported
  
* fast: MIP tries to incurr a minimum overhead layer and directly talks to
  the solver C API using ctypes


