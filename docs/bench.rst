.. _chapBenchmarks:

Benchmarks
==========

This section presents computational experiments measuring **model creation
time** — the time from an empty model to a fully built, solver-ready instance
— across different modelling interfaces and solver backends.

Python-MIP communicates every problem modification directly to the solver
engine rather than staging a separate intermediate model.  To do this
efficiently without per-call overhead:

- **CBC** buffers column and row additions in a CFFI C-array cache inside the
  ``Cbc_C_Interface`` and flushes them in bulk via ``OsiSolverInterface``
  batch calls.
- **HiGHS** uses a matching CFFI C-array cache in ``mip/highs.py`` that
  accumulates pending columns (with their integrality markers) and rows in CSR
  layout, then flushes via ``Highs_addCols`` / ``Highs_addRows``.  Any
  query or modification of a committed row or column triggers an automatic
  flush first.
- **Gurobi** provides its own internal buffering (``update`` mode) that
  python-mip relies on directly.

The ``highspy`` native batch API (``addVars`` / ``addRows`` with numpy arrays)
represents the theoretical lower bound for HiGHS model creation: a single bulk
call with a pre-built CSR matrix, bypassing all Python object overhead.
The ``highspy`` high-level API (``addVariable`` / ``addConstr`` expression
objects) is included for reference.

Experiments were run on CPython 3.14.4 on a Linux workstation.
Reproducible benchmark scripts are in the ``benchmarks/`` directory.


n-Queens
--------

Binary integer programs: place :math:`n` non-attacking queens on an
:math:`n \times n` board.  The model has :math:`n^2` binary variables,
:math:`2n` equality constraints (rows and columns) and up to
:math:`2(2n-3)` at-most-one diagonal constraints.  The :math:`n=500`
instance has 250,000 binary variables.

Model creation times in seconds (CPython 3.14.4):

.. list-table::
   :header-rows: 1
   :align: center
   :widths: 10 16 16 16 18 20

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
     - python-mip / Gurobi
     - highspy (hl API)
     - highspy (batch numpy)
   * - 100
     - 0.037
     - 0.035
     - 0.038
     - 0.312
     - **0.006**
   * - 200
     - 0.147
     - 0.131
     - 0.152
     - 1.141
     - **0.017**
   * - 300
     - 0.347
     - 0.414
     - 0.443
     - 3.100
     - **0.043**
   * - 400
     - 0.708
     - 0.696
     - **0.660**
     - 5.561
     - 0.083
   * - 500
     - **1.165**
     - 1.101
     - 1.173
     - 8.734
     - 0.121

Python-MIP with any backend is **8–25× faster** than the highspy high-level
API for model creation, and within a factor of 10–12 of the highspy batch
numpy API which requires the user to pre-build a full CSR matrix.

Run: ``python benchmarks/queens_bench.py --build-only 100 200 300 400 500``


TSP single-commodity flow
--------------------------

Mixed-integer programs using the compact Gavish–Graves (1978) single-commodity
flow formulation for the Travelling Salesman Problem on random Euclidean
instances (fixed seed for reproducibility):

.. math::

   \min\ \sum_{i \neq j} c_{ij} x_{ij}

   \text{s.t.} \quad
   \sum_j x_{ij} = 1,\quad \sum_i x_{ij} = 1 \quad \forall i, j

   y_{ij} \leq (n-1)\,x_{ij} \quad \forall i \neq j

   \sum_j y_{ji} - \sum_j y_{ij} = 1 \quad \forall i = 1,\ldots,n-1

where :math:`x_{ij} \in \{0,1\}` selects arcs and :math:`y_{ij} \geq 0`
carries flow.  City 0 is the depot; the capacity and flow-conservation
constraints together eliminate all subtours.

For :math:`n` cities the model has :math:`2n(n-1)` variables and
:math:`2n + n(n-1) + (n-1)` constraints.

Model creation times in seconds (CPython 3.14.4):

.. list-table::
   :header-rows: 1
   :align: center
   :widths: 10 16 16 16 18 20

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
     - python-mip / Gurobi
     - highspy (hl API)
     - highspy (batch numpy)
   * - 15
     - 0.006
     - 0.005
     - **0.004**
     - 0.009
     - 0.000
   * - 20
     - 0.005
     - 0.005
     - 0.007
     - 0.024
     - **0.001**
   * - 30
     - 0.011
     - 0.028
     - 0.013
     - 0.039
     - **0.001**
   * - 50
     - **0.032**
     - 0.032
     - 0.037
     - 0.106
     - 0.003
   * - 75
     - 0.076
     - **0.071**
     - 0.076
     - 0.248
     - 0.007
   * - 100
     - 0.141
     - **0.133**
     - 0.145
     - 0.475
     - 0.011
   * - 150
     - 0.349
     - **0.299**
     - 0.341
     - 1.050
     - 0.030

The TSP flow model interleaves binary and continuous variables with
variable-density rows (degree rows touch :math:`n-1` variables; capacity rows
touch 2; flow-conservation rows touch :math:`2(n-1)`).  Python-MIP's cache
handles this automatically — no manual CSR construction required.
Python-MIP is roughly **3× faster** than the highspy high-level API and
within **10–15×** of the highspy batch numpy API that requires the caller to
pre-build the full CSR matrix.

To verify correctness and solve a small instance:
``python benchmarks/tsp_flow_bench.py --verify``

Run: ``python benchmarks/tsp_flow_bench.py --build-only 15 20 30 50 75 100 150``

