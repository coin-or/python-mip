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
:math:`2(2n-3)` at-most-one diagonal constraints.  The :math:`n=1200`
instance has 1,440,000 binary variables.

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
   * - 200
     - **0.105**
     - 0.100
     - 0.126
     - 1.344
     - 0.020
   * - 400
     - 0.496
     - **0.437**
     - 0.544
     - 4.950
     - 0.074
   * - 600
     - **1.104**
     - 1.093
     - 1.221
     - >8s
     - 0.173
   * - 800
     - 2.056
     - **1.991**
     - 2.467
     - >8s
     - 0.325
   * - 1000
     - 3.447
     - **3.348**
     - 3.926
     - >8s
     - 0.527
   * - 1200
     - 5.120
     - **4.844**
     - 5.881
     - >8s
     - 0.826

Python-MIP with any backend is **10–12× faster** than the highspy high-level
API for model creation (highspy-hl times out above n=400 with an 8 s build
limit), and within a factor of 6–7 of the highspy batch numpy API which
requires the user to pre-build a full CSR matrix.

Run: ``python benchmarks/queens_bench.py --build-only``


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
   * - 30
     - **0.012**
     - 0.011
     - 0.013
     - 0.041
     - 0.001
   * - 50
     - 0.033
     - **0.027**
     - 0.034
     - 0.113
     - 0.003
   * - 75
     - 0.074
     - **0.065**
     - 0.077
     - 0.262
     - 0.007
   * - 100
     - 0.134
     - **0.120**
     - 0.171
     - 0.500
     - 0.013
   * - 150
     - **0.322**
     - 0.342
     - 0.370
     - 1.248
     - 0.036
   * - 200
     - **0.663**
     - 0.569
     - 0.664
     - 2.547
     - 0.065
   * - 300
     - 1.449
     - **1.333**
     - 1.541
     - 5.972
     - 0.167
   * - 400
     - 2.667
     - **2.501**
     - 2.798
     - >8s
     - 0.290
   * - 500
     - 3.757
     - **3.358**
     - 4.557
     - >8s
     - 0.503

The TSP flow model interleaves binary and continuous variables with
variable-density rows (degree rows touch :math:`n-1` variables; capacity rows
touch 2; flow-conservation rows touch :math:`2(n-1)`).  Python-MIP's cache
handles this automatically — no manual CSR construction required.
Python-MIP is roughly **3–5× faster** than the highspy high-level API (which
times out above n=300 with an 8 s build limit) and within **7–8×** of the
highspy batch numpy API that requires the caller to pre-build the full CSR
matrix.

To verify correctness and solve a small instance:
``python benchmarks/tsp_flow_bench.py --verify``

Run: ``python benchmarks/tsp_flow_bench.py --build-only``

