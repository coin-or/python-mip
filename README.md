# Python MIP (Mixed-Integer Linear Programming) Tools

Python MIP is a collection of tools for python the modelling and solution
of Mixed-Integer linear programs. MIP syntax was inspired in Pulp and
porting Pulp models should be quite easy.

Some of the main features of MIP are:

* multi solver: works with many different MIP solvers, such as Gurobi, CBC
  and GLPK
* fast: MIP tries to incurr a minimum overhead layer and directly talks to
  the solver C API using ctypes


