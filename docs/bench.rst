.. _chapBenchmarks:

Benchmarks
==========

This section presents computational experiments on the creation of Integer
Programming models using different mathematical modelling packages. Gurobi is
the default Gurobi :sup:`®` Python interface, which currently only supports the
Python interpreter CPython. Pulp supports both CPython and also the
just-in-time compiler Pypy. MIP also suports both. JuMP [DHL17I]_ is the
Mathematical Programming package of the Julia programming language. Both
Jump and Pulp use intermediate data structures to store the mathematical
programming model before flushing it to the solver, so that the selected
solver does not impacts on the model creation times. MIP does not stores
the model itself, directly calling problem creation/modification routines
on the solver engine.

Since MIP communicates every problem modification directly to the solver
engine, the engine must handle efficiently many small modification request
to avoid potentially expensive resize/move operations in the constraint
matrix. Gurobi automatically buffers problem modification requests and has
an update method to flush these request. CBC did not had an equivalent
mechanism, so we implemented an automatic buffering/flushing mechanism in
the CBC C Interface. Our computational results already consider this
updated CBC version.

Computational experiments executed on a ThinkPad :sup:`®` X1 notebook with
an Intel :sup:`®` Core™ i7-7600U processor and 8 Gb of RAM using the
Linux operating system. The following software releases were used: CPython
3.7.3, Pypy 7.1.1, Julia 1.1.1, JuMP 0.19 and Gurobi 8.1 and CBC svn 2563.


n-Queens
--------

These are binary programming models. The largest model has 1 million
variables and roughly 6000 constraints and 4 million of non-zero entries
in the constraint matrix.


  +------------+------------+------------+-----------+-----------------------+-----------------------+-----------+
  |            |            |         Pulp           |                  Python-MIP                   |           |
  |            |   Gurobi   |                        +-----------------------+-----------------------+           |
  |            |            |                        |        Gurobi         |         CBC           |   JuMP    |
  |            |   CPython  +------------+-----------+------------+----------+------------+----------+           |
  |  :math:`n` |            |   CPython  |   Pypy    |   CPython  |   Pypy   |   CPython  |   Pypy   |           |
  +------------+------------+------------+-----------+------------+----------+------------+----------+-----------+
  |        100 |       0.24 |       0.27 |  **0.18** |       0.65 |     0.97 |       0.30 |     0.45 |      2.38 |
  +------------+------------+------------+-----------+------------+----------+------------+----------+-----------+
  |        200 |       1.43 |       1.73 |      0.43 |       1.60 | **0.18** |       1.47 |     0.19 |      0.25 |
  +------------+------------+------------+-----------+------------+----------+------------+----------+-----------+
  |        300 |       4.97 |       5.80 |      0.92 |       5.48 | **0.37** |       5.12 |     0.38 |      0.47 |
  +------------+------------+------------+-----------+------------+----------+------------+----------+-----------+
  |        400 |      12.37 |      14.29 |      1.55 |      13.58 | **0.72** |      13.24 |     0.74 |      1.11 |
  +------------+------------+------------+-----------+------------+----------+------------+----------+-----------+
  |        500 |      24.70 |      32.62 |      2.62 |      27.25 |     1.25 |      26.30 | **1.23** |      2.09 |
  +------------+------------+------------+-----------+------------+----------+------------+----------+-----------+
  |        600 |      43.88 |      49.92 |      4.10 |      47.75 |     2.02 |      46.23 | **1.99** |      2.86 |
  +------------+------------+------------+-----------+------------+----------+------------+----------+-----------+
  |        700 |      69.77 |      79.57 |      5.84 |      75.97 |     3.04 |      74.47 | **2.94** |      3.64 |
  +------------+------------+------------+-----------+------------+----------+------------+----------+-----------+
  |        800 |     105.04 |     119.07 |      8.19 |     114.86 |     4.33 |     112.10 | **4.26** |      5.58 |
  +------------+------------+------------+-----------+------------+----------+------------+----------+-----------+
  |        900 |     150.89 |     169.92 |     10.84 |     163.36 |     5.95 |     160.67 | **5.83** |      8.08 |
  +------------+------------+------------+-----------+------------+----------+------------+----------+-----------+
  |       1000 |     206.63 |     232.32 |     14.26 |     220.56 |     8.02 |     222.09 | **7.76** |     10.02 |
  +------------+------------+------------+-----------+------------+----------+------------+----------+-----------+


