.. _chapBenchmarks:

Benchmarks
==========

This section presents computational experiments measuring **model creation
time** — the time from an empty model to a fully built, solver-ready instance
— across different modelling interfaces, solver backends, and Python
interpreters.

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

Both CBC and HiGHS use CFFI and are fully **PyPy-compatible**.  Gurobi's
``gurobipy`` extension is also compatible with PyPy.  The PyPy JIT compiler
eliminates most Python overhead, yielding 4–5× faster model creation times.

The ``highspy`` native batch API (``addVars`` / ``addRows`` with numpy arrays)
represents the theoretical lower bound for HiGHS model creation: a single bulk
call with a pre-built CSR matrix, bypassing all Python object overhead.
The ``highspy`` high-level API (``addVariable`` / ``addConstr`` expression
objects) is included for reference.

Experiments were run on a Linux workstation.
Reproducible benchmark scripts are in the ``benchmarks/`` directory.


n-Queens
--------

See :ref:`queens-label` for the problem formulation and example code.
The :math:`n=1200` instance has 1,440,000 binary variables, 2,400 equality
constraints, and up to 4,794 at-most-one diagonal constraints.

Model creation times in seconds — **CPython 3.14.4**:

.. list-table::
   :header-rows: 1
   :align: center
   :widths: 8 12 12 13 11 14 16

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
     - python-mip / Gurobi
     - gurobipy
     - highspy (hl API)
     - highspy (batch numpy)
   * - 200
     - 0.150
     - **0.134**
     - 0.160
     - 0.224
     - 1.739
     - 0.025
   * - 400
     - 0.710
     - **0.680**
     - 0.791
     - 1.022
     - 7.827
     - 0.096
   * - 600
     - 1.631
     - **1.575**
     - 1.940
     - 2.324
     - >8s
     - 0.248
   * - 800
     - 3.031
     - **2.749**
     - 3.224
     - 4.353
     - >8s
     - 0.485
   * - 1000
     - 4.847
     - **4.170**
     - 5.431
     - 6.974
     - >8s
     - 0.677
   * - 1200
     - 6.687
     - **6.271**
     - 7.513
     - >8s
     - >8s
     - 1.146

Model creation times in seconds — **PyPy 3.11 (7.3.20)**:

.. list-table::
   :header-rows: 1
   :align: center
   :widths: 10 16 16 16

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
     - python-mip / Gurobi
   * - 200
     - 0.061
     - **0.053**
     - 0.050
   * - 400
     - **0.141**
     - 0.153
     - 0.189
   * - 600
     - **0.274**
     - 0.290
     - 0.371
   * - 800
     - **0.471**
     - 0.463
     - 0.684
   * - 1000
     - **0.774**
     - 0.862
     - 1.179
   * - 1200
     - **1.100**
     - 1.117
     - 1.620

PyPy delivers a **4–5× speedup** over CPython for python-mip model building.
The highspy batch numpy API is slower under PyPy (numpy operations are not
JIT-compiled by PyPy) and is omitted from the PyPy table.

Python-MIP (CPython) with CBC or HiGHS is **1.3–1.7× faster** than
raw ``gurobipy`` at large sizes, and **10–12× faster** than the highspy
high-level API (which times out above n=400 with an 8 s build limit), and
within a factor of 6–7 of the highspy batch numpy API which requires the user
to pre-build a full CSR matrix.

Run: ``python benchmarks/queens_bench.py --build-only``


TSP single-commodity flow
--------------------------

Mixed-integer programs using the compact [GaGr78]_ single-commodity
flow formulation for the Travelling Salesman Problem (see
:ref:`tsp-label` for the formulation and example code) on random Euclidean
instances (fixed seed for reproducibility).

For :math:`n` cities the model has :math:`2n(n-1)` variables and
:math:`2n + n(n-1) + (n-1)` constraints.

Model creation times in seconds — **CPython 3.14.4**:

.. list-table::
   :header-rows: 1
   :align: center
   :widths: 8 12 12 13 11 14 16

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
     - python-mip / Gurobi
     - gurobipy
     - highspy (hl API)
     - highspy (batch numpy)
   * - 30
     - 0.031
     - **0.029**
     - 0.038
     - 0.036
     - 0.100
     - 0.004
   * - 50
     - 0.080
     - 0.078
     - 0.063
     - **0.061**
     - 0.213
     - 0.006
   * - 75
     - 0.150
     - **0.132**
     - 0.145
     - 0.220
     - 0.659
     - 0.014
   * - 100
     - **0.281**
     - 0.283
     - 0.394
     - 0.296
     - 0.959
     - 0.043
   * - 150
     - 0.730
     - **0.524**
     - 0.564
     - 0.732
     - 2.086
     - 0.048
   * - 200
     - 1.083
     - **0.930**
     - 1.128
     - 1.480
     - 4.408
     - 0.157
   * - 300
     - 3.092
     - **2.283**
     - 2.823
     - 2.936
     - >8s
     - 0.315
   * - 400
     - 5.182
     - 5.152
     - 5.299
     - **4.909**
     - >8s
     - 0.513
   * - 500
     - >8s
     - **6.909**
     - >8s
     - >8s
     - >8s
     - 1.037

Model creation times in seconds — **PyPy 3.11 (7.3.20)**:

.. list-table::
   :header-rows: 1
   :align: center
   :widths: 10 16 16 16

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
     - python-mip / Gurobi
   * - 75
     - **0.047**
     - 0.030
     - 0.020
   * - 100
     - **0.028**
     - 0.031
     - 0.062
   * - 150
     - **0.101**
     - 0.116
     - 0.110
   * - 200
     - **0.159**
     - 0.193
     - 0.195
   * - 300
     - **0.383**
     - 0.433
     - 0.469
   * - 400
     - **0.634**
     - 0.814
     - 0.821
   * - 500
     - **1.263**
     - 1.224
     - 1.345

PyPy delivers a **3–4× speedup** over CPython for the TSP flow model.  At
small sizes (:math:`n \leq 50`) JIT warm-up may exceed CPython; the benefit is clear
from n=75 onwards.

The TSP flow model interleaves binary and continuous variables with
variable-density rows (degree rows touch :math:`n-1` variables; capacity rows
touch 2; flow-conservation rows touch :math:`2(n-1)`).  Python-MIP's cache
handles this automatically — no manual CSR construction required.
Python-MIP (CPython) with HiGHS is roughly on par with ``gurobipy`` at most
sizes and **3–5× faster** than the highspy high-level API (which times out
above n=300 with an 8 s build limit) and within **7–8×** of the highspy batch
numpy API that requires the caller to pre-build the full CSR matrix.

To verify correctness and solve a small instance:
``python benchmarks/tsp_flow_bench.py --verify``

Run: ``python benchmarks/tsp_flow_bench.py --build-only``


Resource-Constrained Project Scheduling (RCPSP)
------------------------------------------------

Binary integer programs based on the [PWW69]_ time-indexed formulation
(see :ref:`rcpsp-label` for the formulation and example code).
For :math:`n` jobs with processing times in :math:`[1,4]`, the time
horizon :math:`T = \sum_j p_j` grows linearly with :math:`n`, so the
model has :math:`O(n^2)` binary variables overall.

Random instances use processing times drawn from :math:`[1,4]` and are
tested in four configurations combining two values for the number of
resources (2 or 4) and two values for the number of extra precedence arcs
among real jobs (sparse: :math:`n` arcs, dense: :math:`3n` arcs).

.. rubric:: Config A — 2 resources, sparse precedences

Model creation times in seconds — **CPython 3.14.4**:

.. list-table::
   :header-rows: 1
   :align: center
   :widths: 10 16 16 16 16

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
     - python-mip / Gurobi
     - gurobipy
   * - 10
     - 0.084
     - **0.013**
     - 0.036
     - 0.013
   * - 20
     - 0.046
     - 0.047
     - 0.045
     - **0.042**
   * - 30
     - 0.108
     - 0.116
     - 0.070
     - **0.063**
   * - 50
     - **0.207**
     - 0.237
     - 0.239
     - 0.295
   * - 75
     - 0.667
     - 0.588
     - 0.580
     - **0.536**
   * - 100
     - 1.222
     - 0.997
     - 1.057
     - **0.974**
   * - 150
     - 2.329
     - 2.665
     - 2.654
     - **2.180**
   * - 200
     - 4.945
     - 4.824
     - 4.483
     - **3.694**

Model creation times in seconds — **PyPy 3.11 (7.3.20)**:

.. list-table::
   :header-rows: 1
   :align: center
   :widths: 10 16 16

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
   * - 30
     - **0.026**
     - 0.032
   * - 50
     - **0.053**
     - 0.060
   * - 75
     - **0.119**
     - 0.143
   * - 100
     - **0.196**
     - 0.241
   * - 150
     - 0.509
     - **0.464**
   * - 200
     - **0.810**
     - 0.855

.. rubric:: Config B — 2 resources, dense precedences

Model creation times in seconds — **CPython 3.14.4**:

.. list-table::
   :header-rows: 1
   :align: center
   :widths: 10 16 16 16 16

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
     - python-mip / Gurobi
     - gurobipy
   * - 10
     - 0.013
     - **0.011**
     - 0.011
     - 0.012
   * - 20
     - 0.041
     - 0.045
     - 0.046
     - **0.062**
   * - 30
     - 0.098
     - 0.120
     - **0.097**
     - 0.125
   * - 50
     - 0.283
     - 0.278
     - **0.273**
     - 0.274
   * - 75
     - 0.887
     - **0.719**
     - 0.917
     - 0.708
   * - 100
     - 1.922
     - **1.708**
     - 1.477
     - 1.843
   * - 150
     - 3.996
     - 3.687
     - **3.113**
     - 3.208
   * - 200
     - 5.621
     - 5.724
     - 7.342
     - **5.686**

Model creation times in seconds — **PyPy 3.11 (7.3.20)**:

.. list-table::
   :header-rows: 1
   :align: center
   :widths: 10 16 16

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
   * - 30
     - **0.018**
     - 0.025
   * - 50
     - 0.077
     - **0.060**
   * - 75
     - 0.155
     - **0.134**
   * - 100
     - 0.342
     - **0.248**
   * - 150
     - **0.578**
     - 0.674
   * - 200
     - **1.109**
     - 1.139

.. rubric:: Config C — 4 resources, sparse precedences

Model creation times in seconds — **CPython 3.14.4**:

.. list-table::
   :header-rows: 1
   :align: center
   :widths: 10 16 16 16 16

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
     - python-mip / Gurobi
     - gurobipy
   * - 10
     - 0.017
     - **0.017**
     - 0.018
     - 0.022
   * - 20
     - 0.070
     - 0.070
     - 0.055
     - **0.032**
   * - 30
     - **0.085**
     - 0.096
     - 0.162
     - 0.133
   * - 50
     - 0.367
     - **0.279**
     - 0.440
     - 0.358
   * - 75
     - 0.880
     - 1.030
     - 0.817
     - **0.690**
   * - 100
     - 1.768
     - 1.630
     - 1.636
     - **1.155**
   * - 150
     - 3.902
     - 4.047
     - 3.491
     - **2.686**
   * - 200
     - 4.949
     - 4.955
     - 5.428
     - **3.800**

Model creation times in seconds — **PyPy 3.11 (7.3.20)**:

.. list-table::
   :header-rows: 1
   :align: center
   :widths: 10 16 16

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
   * - 30
     - 0.020
     - **0.017**
   * - 50
     - 0.050
     - **0.046**
   * - 75
     - **0.128**
     - 0.241
   * - 100
     - **0.239**
     - 0.269
   * - 150
     - 0.728
     - **0.664**
   * - 200
     - 1.144
     - **1.060**

.. rubric:: Config D — 4 resources, dense precedences

Model creation times in seconds — **CPython 3.14.4**:

.. list-table::
   :header-rows: 1
   :align: center
   :widths: 10 16 16 16 16

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
     - python-mip / Gurobi
     - gurobipy
   * - 10
     - 0.011
     - **0.011**
     - 0.015
     - 0.022
   * - 20
     - 0.046
     - **0.043**
     - 0.048
     - 0.042
   * - 30
     - 0.122
     - 0.105
     - 0.114
     - **0.092**
   * - 50
     - 0.308
     - 0.326
     - 0.303
     - **0.264**
   * - 75
     - 0.762
     - 0.888
     - 0.798
     - **0.699**
   * - 100
     - 1.428
     - 1.523
     - 1.543
     - **1.171**
   * - 150
     - 3.228
     - 3.269
     - 3.092
     - **2.729**
   * - 200
     - 5.712
     - 5.640
     - 5.573
     - **4.841**

Model creation times in seconds — **PyPy 3.11 (7.3.20)**:

.. list-table::
   :header-rows: 1
   :align: center
   :widths: 10 16 16

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
   * - 30
     - 0.021
     - **0.018**
   * - 50
     - **0.054**
     - 0.061
   * - 75
     - 0.313
     - **0.162**
   * - 100
     - **0.318**
     - 0.368
   * - 150
     - **0.705**
     - 0.793
   * - 200
     - **1.284**
     - 1.365

PyPy delivers a **4–5× speedup** across all RCPSP configurations.  The
RCPSP model is structurally the most complex of the three benchmarks: it
combines binary variables with dense resource-capacity constraints (each
touching :math:`O(n \cdot \bar{p})` non-zeros) and quadratic scaling in
problem size.  Python-MIP's automatic flush policy correctly handles the
interleaved variable and constraint additions without any manual CSR
construction.  ``gurobipy`` is consistently **10–25% faster** than
python-mip/Gurobi at large sizes in this benchmark, reflecting the overhead
of the python-mip CFFI abstraction layer for Gurobi; CBC and HiGHS are
competitive with ``gurobipy`` in most configurations.

Run: ``python benchmarks/rcpsp_bench.py --build-only``

