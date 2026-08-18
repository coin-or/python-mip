.. _chapBenchmarks:

Benchmarks
==========

This section presents computational experiments measuring **model creation
time** — the time from an empty model to a fully built, solver-ready instance
— across different modelling interfaces, solver backends, and Python
interpreters.

Python-MIP communicates every problem modification directly to the solver
engine rather than staging a separate intermediate model, buffering column
and row additions internally (CFFI C-array caches for CBC and HiGHS,
Gurobi's own ``update`` mode) to avoid per-call overhead.

CBC, HiGHS and Gurobi are all accessed through CFFI bindings to the solvers'
native C APIs, and are therefore fully **PyPy-compatible**. ``gurobipy``
itself does **not** ship a PyPy wheel — python-mip's CFFI-based Gurobi
backend is the only way to drive Gurobi from PyPy, yielding 2.5–5× faster
model creation times there depending on model structure (see below).

We compare against raw ``gurobipy`` and ``highspy`` (HiGHS's own Python
package), including ``highspy``'s recommended vectorized numpy-array API
and its lowest-level batch API (a pre-built CSR matrix, the fastest option
in every benchmark but with no incremental modelling interface). See
:ref:`bench-appendix-clarity` for a discussion of these trade-offs with
side-by-side code.

Experiments were run on a Linux workstation.
Reproducible benchmark scripts are in the ``benchmarks/`` directory.


n-Queens
--------

See :ref:`queens-label` for the problem formulation and example code.
The :math:`n=1200` instance has 1,440,000 binary variables, 2,400 equality
constraints, and up to 4,794 at-most-one diagonal constraints.

The ``highspy`` "high-level API" column uses HiGHS's own recommended
vectorized numpy-array style (``addBinaries``/``addConstrs`` with
``.sum(axis=...)``/``.diagonal()``), matching the pattern used in HiGHS's
own ``examples/nqueens.py`` — not a naive per-variable/per-constraint loop.

Model creation times in seconds — **CPython 3.14.4**:

.. tabularcolumns:: rrrrrrr

.. list-table::
   :header-rows: 1
   :align: center
   :class: numtable
   :widths: 8 12 12 13 11 14 16

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
     - python-mip / Gurobi
     - gurobipy
     - highspy (hl API)
     - highspy (batch numpy)
   * - 200
     - 0.305
     - **0.252**
     - 0.305
     - 0.380
     - 0.268
     - 0.040
   * - 400
     - 1.348
     - **1.100**
     - 1.272
     - 1.623
     - 1.073
     - 0.164
   * - 600
     - 3.162
     - 2.288
     - 3.139
     - 3.725
     - **2.671**
     - 0.375
   * - 800
     - 5.417
     - **4.430**
     - 4.953
     - 7.001
     - 5.046
     - 0.702
   * - 1000
     - >8s
     - **7.087**
     - 7.752
     - >8s
     - >8s
     - 1.082
   * - 1200
     - >8s
     - >8s
     - >8s
     - >8s
     - >8s
     - **1.559**

Model creation times in seconds — **PyPy 3.11 (7.3.20)**:

``gurobipy`` has no PyPy wheel and is omitted. ``highspy`` (both hl and
batch APIs) relies heavily on numpy, which PyPy does not JIT-compile —
both are included here for completeness but are not representative of
PyPy's strengths.

.. tabularcolumns:: rrrrrr

.. list-table::
   :header-rows: 1
   :align: center
   :class: numtable
   :widths: 8 12 12 13 14 16

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
     - python-mip / Gurobi
     - highspy (hl API)
     - highspy (batch numpy)
   * - 200
     - 0.163
     - 0.193
     - **0.139**
     - 0.820
     - 0.230
   * - 400
     - 0.592
     - **0.467**
     - 0.521
     - 2.257
     - 0.553
   * - 600
     - 1.357
     - 0.872
     - **0.791**
     - 5.315
     - 1.107
   * - 800
     - 2.233
     - 1.574
     - **1.529**
     - >8s
     - 2.183
   * - 1000
     - 3.651
     - 2.261
     - **2.465**
     - >8s
     - 3.355
   * - 1200
     - 5.077
     - 3.619
     - **3.486**
     - >8s
     - 4.447

PyPy delivers a **2–4× speedup** over CPython for python-mip model building
(smaller than earlier measurements taken on a less loaded machine, but the
same direction and order of magnitude).

Once ``highspy``'s vectorized API is used fairly, the picture for CPython is
much closer than a naive-API comparison would suggest: python-mip/HiGHS is
still the fastest single-variable-at-a-time backend at most sizes, roughly
on par with (or a little ahead of) both ``highspy``'s vectorized high-level
API and raw ``gurobipy``. The ``highspy`` batch numpy API remains far ahead
of everything else, as expected for a hand-built CSR bulk call. Under PyPy,
``highspy``'s reliance on numpy works against it — both its APIs degrade
sharply, while CBC/HiGHS/Gurobi through python-mip keep scaling smoothly.

Run: ``python benchmarks/queens_bench.py --build-only``


TSP single-commodity flow
--------------------------

Mixed-integer programs using the compact [GaGr78]_ single-commodity
flow formulation for the Travelling Salesman Problem (see
:ref:`tsp-label` for the formulation and example code) on random Euclidean
instances (fixed seed for reproducibility).

For :math:`n` cities the model has :math:`2n(n-1)` variables and
:math:`2n + n(n-1) + (n-1)` constraints.

The ``highspy`` "high-level API" column here uses HiGHS's vectorized
numpy-array API too, but this problem only has arcs :math:`i \neq j` — a
genuinely sparse index set. To use the dense array API at all, the diagonal
entries have to be allocated as real (unused) variables and then fixed to 0
via ``changeColBounds`` after the fact. That workaround, and the resulting
:math:`O(n^2)` dense elementwise arithmetic instead of the natural
:math:`O(n(n-1))` sparse arc list, is exactly the caveat we expected: it
works, but scales worse than the naive per-arc loop once :math:`n` grows,
and the code is less direct than either python-mip's dict-of-arcs style or
the naive ``highspy`` loop.

Model creation times in seconds — **CPython 3.14.4**:

.. tabularcolumns:: rrrrrrr

.. list-table::
   :header-rows: 1
   :align: center
   :class: numtable
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
     - **0.026**
     - 0.034
     - 0.031
     - 0.031
     - 0.003
   * - 50
     - 0.074
     - **0.066**
     - 0.077
     - 0.091
     - 0.077
     - 0.007
   * - 75
     - 0.169
     - **0.152**
     - 0.172
     - 0.205
     - 0.201
     - 0.016
   * - 100
     - 0.356
     - **0.273**
     - 0.332
     - 0.369
     - 0.331
     - 0.042
   * - 150
     - 0.811
     - **0.632**
     - 0.698
     - 0.804
     - 0.864
     - 0.074
   * - 200
     - 1.329
     - **1.204**
     - 1.244
     - 1.455
     - 1.803
     - 0.133
   * - 300
     - 3.131
     - **2.632**
     - 3.249
     - 3.257
     - 5.813
     - 0.399
   * - 400
     - 5.923
     - **4.903**
     - 5.636
     - 6.147
     - >8s
     - 0.661
   * - 500
     - >8s
     - >8s
     - >8s
     - >8s
     - >8s
     - **1.176**

Model creation times in seconds — **PyPy 3.11 (7.3.20)**:

``gurobipy`` has no PyPy wheel and is omitted.

.. tabularcolumns:: rrrrrr

.. list-table::
   :header-rows: 1
   :align: center
   :class: numtable
   :widths: 8 12 12 13 14 16

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
     - python-mip / Gurobi
     - highspy (hl API)
     - highspy (batch numpy)
   * - 30
     - 0.098
     - 0.093
     - **0.103**
     - 0.231
     - 0.014
   * - 50
     - 0.106
     - **0.103**
     - 0.125
     - 0.418
     - 0.050
   * - 75
     - 0.104
     - **0.090**
     - 0.118
     - 0.654
     - 0.106
   * - 100
     - 0.115
     - **0.074**
     - 0.156
     - 1.197
     - 0.139
   * - 150
     - 0.249
     - **0.171**
     - 0.273
     - 2.472
     - 0.433
   * - 200
     - 0.448
     - 0.591
     - **0.304**
     - 4.556
     - 0.816
   * - 300
     - 1.180
     - 1.293
     - **1.099**
     - >8s
     - 1.541
   * - 400
     - 2.110
     - **1.703**
     - 1.703
     - >8s
     - 2.748
   * - 500
     - 3.665
     - 3.523
     - **2.656**
     - >8s
     - 4.315

PyPy delivers a solid speedup over CPython for python-mip's three backends
here too (roughly 1.5–3× depending on size), while the numpy-heavy
``highspy`` hl variant is markedly *worse* under PyPy than on CPython — the
same pattern seen in n-Queens, amplified by the wasted dense diagonal here.

The TSP flow model interleaves binary and continuous variables with
variable-density rows (degree rows touch :math:`n-1` variables; capacity rows
touch 2; flow-conservation rows touch :math:`2(n-1)`).  Python-MIP's cache
handles this automatically — no manual CSR construction, and no dense-array
workaround for the sparse arc set, required. python-mip/HiGHS is
consistently the fastest incremental-API backend here, ahead of
``gurobipy`` and both python-mip/CBC and python-mip/Gurobi at most sizes;
``highspy``'s vectorized hl API is competitive up to n ~ 150 but degrades
faster than every other backend beyond that, exactly where its dense-array
requirement stops fitting this sparse problem. The batch numpy API remains
comfortably the fastest, and the only backend that completes n=500 within
the 8 s budget.

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

Model creation times in seconds — **CPython 3.14.4** (Res = number of resources,
Prec = precedence density):

.. tabularcolumns:: rllrrrr

.. list-table::
   :header-rows: 1
   :align: center
   :class: numtable rcpsp-table
   :widths: 6 6 8 14 14 16 14

   * - :math:`n`
     - Res
     - Prec
     - python-mip / CBC
     - python-mip / HiGHS
     - python-mip / Gurobi
     - gurobipy
   * - 10
     - 2
     - sparse
     - 0.084
     - **0.013**
     - 0.036
     - 0.013
   * - 20
     - 2
     - sparse
     - 0.046
     - 0.047
     - 0.045
     - **0.042**
   * - 30
     - 2
     - sparse
     - 0.108
     - 0.116
     - 0.070
     - **0.063**
   * - 50
     - 2
     - sparse
     - **0.207**
     - 0.237
     - 0.239
     - 0.295
   * - 75
     - 2
     - sparse
     - 0.667
     - 0.588
     - 0.580
     - **0.536**
   * - 100
     - 2
     - sparse
     - 1.222
     - 0.997
     - 1.057
     - **0.974**
   * - 150
     - 2
     - sparse
     - 2.329
     - 2.665
     - 2.654
     - **2.180**
   * - 200
     - 2
     - sparse
     - 4.945
     - 4.824
     - 4.483
     - **3.694**
   * - 10
     - 2
     - dense
     - 0.013
     - **0.011**
     - 0.011
     - 0.012
   * - 20
     - 2
     - dense
     - 0.041
     - 0.045
     - 0.046
     - **0.062**
   * - 30
     - 2
     - dense
     - 0.098
     - 0.120
     - **0.097**
     - 0.125
   * - 50
     - 2
     - dense
     - 0.283
     - 0.278
     - **0.273**
     - 0.274
   * - 75
     - 2
     - dense
     - 0.887
     - **0.719**
     - 0.917
     - 0.708
   * - 100
     - 2
     - dense
     - 1.922
     - **1.708**
     - 1.477
     - 1.843
   * - 150
     - 2
     - dense
     - 3.996
     - 3.687
     - **3.113**
     - 3.208
   * - 200
     - 2
     - dense
     - 5.621
     - 5.724
     - 7.342
     - **5.686**
   * - 10
     - 4
     - sparse
     - 0.017
     - **0.017**
     - 0.018
     - 0.022
   * - 20
     - 4
     - sparse
     - 0.070
     - 0.070
     - 0.055
     - **0.032**
   * - 30
     - 4
     - sparse
     - **0.085**
     - 0.096
     - 0.162
     - 0.133
   * - 50
     - 4
     - sparse
     - 0.367
     - **0.279**
     - 0.440
     - 0.358
   * - 75
     - 4
     - sparse
     - 0.880
     - 1.030
     - 0.817
     - **0.690**
   * - 100
     - 4
     - sparse
     - 1.768
     - 1.630
     - 1.636
     - **1.155**
   * - 150
     - 4
     - sparse
     - 3.902
     - 4.047
     - 3.491
     - **2.686**
   * - 200
     - 4
     - sparse
     - 4.949
     - 4.955
     - 5.428
     - **3.800**
   * - 10
     - 4
     - dense
     - 0.011
     - **0.011**
     - 0.015
     - 0.022
   * - 20
     - 4
     - dense
     - 0.046
     - **0.043**
     - 0.048
     - 0.042
   * - 30
     - 4
     - dense
     - 0.122
     - 0.105
     - 0.114
     - **0.092**
   * - 50
     - 4
     - dense
     - 0.308
     - 0.326
     - 0.303
     - **0.264**
   * - 75
     - 4
     - dense
     - 0.762
     - 0.888
     - 0.798
     - **0.699**
   * - 100
     - 4
     - dense
     - 1.428
     - 1.523
     - 1.543
     - **1.171**
   * - 150
     - 4
     - dense
     - 3.228
     - 3.269
     - 3.092
     - **2.729**
   * - 200
     - 4
     - dense
     - 5.712
     - 5.640
     - 5.573
     - **4.841**

Model creation times in seconds — **PyPy 3.11 (7.3.20)**:

.. tabularcolumns:: rllrr

.. list-table::
   :header-rows: 1
   :align: center
   :class: numtable rcpsp-table
   :widths: 6 6 8 16 16

   * - :math:`n`
     - Res
     - Prec
     - python-mip / CBC
     - python-mip / HiGHS
   * - 30
     - 2
     - sparse
     - **0.026**
     - 0.032
   * - 50
     - 2
     - sparse
     - **0.053**
     - 0.060
   * - 75
     - 2
     - sparse
     - **0.119**
     - 0.143
   * - 100
     - 2
     - sparse
     - **0.196**
     - 0.241
   * - 150
     - 2
     - sparse
     - 0.509
     - **0.464**
   * - 200
     - 2
     - sparse
     - **0.810**
     - 0.855
   * - 30
     - 2
     - dense
     - **0.018**
     - 0.025
   * - 50
     - 2
     - dense
     - 0.077
     - **0.060**
   * - 75
     - 2
     - dense
     - 0.155
     - **0.134**
   * - 100
     - 2
     - dense
     - 0.342
     - **0.248**
   * - 150
     - 2
     - dense
     - **0.578**
     - 0.674
   * - 200
     - 2
     - dense
     - **1.109**
     - 1.139
   * - 30
     - 4
     - sparse
     - 0.020
     - **0.017**
   * - 50
     - 4
     - sparse
     - 0.050
     - **0.046**
   * - 75
     - 4
     - sparse
     - **0.128**
     - 0.241
   * - 100
     - 4
     - sparse
     - **0.239**
     - 0.269
   * - 150
     - 4
     - sparse
     - 0.728
     - **0.664**
   * - 200
     - 4
     - sparse
     - 1.144
     - **1.060**
   * - 30
     - 4
     - dense
     - 0.021
     - **0.018**
   * - 50
     - 4
     - dense
     - **0.054**
     - 0.061
   * - 75
     - 4
     - dense
     - 0.313
     - **0.162**
   * - 100
     - 4
     - dense
     - **0.318**
     - 0.368
   * - 150
     - 4
     - dense
     - **0.705**
     - 0.793
   * - 200
     - 4
     - dense
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


Capacitated Facility Location
------------------------------

Random Euclidean instances of the classic capacitated facility location
problem (CFLP) — an applied supply-chain / warehouse-location problem, not
a puzzle.  For :math:`n` facilities and :math:`n` customers the model has
:math:`n^2 + n` variables (an :math:`n \times n` continuous assignment
matrix plus :math:`n` binary open/close variables) and only :math:`2n`
constraints (one demand row per customer, one aggregated capacity row per
facility) — the same "many variables, few constraints" shape as n-Queens,
but with dense rows: each demand row touches :math:`n` variables and each
capacity row touches :math:`n+1` variables, and the objective itself sums
:math:`n^2` terms.

Model creation times in seconds — **CPython 3.14.4**:

The ``highspy`` "high-level API" column uses the vectorized numpy-array
API (``addBinaries``/``addVariables``/``addConstrs``) rather than a naive
per-variable loop. CFLP's dense :math:`n \times n` assignment structure
turns out to be an excellent fit for this style — the capacity/demand rows
are natural ``.sum(axis=...)`` reductions over a genuinely dense matrix (no
diagonal-masking workaround needed, unlike TSP).

.. tabularcolumns:: rrrrrrr

.. list-table::
   :header-rows: 1
   :align: center
   :class: numtable
   :widths: 8 12 12 13 11 14 16

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
     - python-mip / Gurobi
     - gurobipy
     - highspy (hl API)
     - highspy (batch numpy)
   * - 200
     - 0.639
     - 0.578
     - 0.603
     - 0.416
     - **0.264**
     - 0.031
   * - 400
     - 2.786
     - 2.451
     - 2.579
     - 1.811
     - **1.021**
     - 0.159
   * - 600
     - 6.790
     - 6.194
     - 6.612
     - 4.092
     - **2.473**
     - 0.389
   * - 800
     - >8s
     - >8s
     - >8s
     - >8s
     - **3.883**
     - 0.723
   * - 1000
     - >8s
     - >8s
     - >8s
     - >8s
     - **6.442**
     - 1.321
   * - 1200
     - >8s
     - >8s
     - >8s
     - >8s
     - >8s
     - **1.692**

Model creation times in seconds — **PyPy 3.11 (7.3.20)**:

``gurobipy`` has no PyPy wheel, so it is omitted from this table — the
python-mip / Gurobi column is the *only* way to drive Gurobi from PyPy.

.. tabularcolumns:: rrrrrr

.. list-table::
   :header-rows: 1
   :align: center
   :class: numtable
   :widths: 8 12 12 13 14 16

   * - :math:`n`
     - python-mip / CBC
     - python-mip / HiGHS
     - python-mip / Gurobi
     - highspy (hl API)
     - highspy (batch numpy)
   * - 200
     - 0.310
     - 0.248
     - **0.238**
     - 0.562
     - 0.203
   * - 400
     - 1.002
     - **0.950**
     - 0.970
     - 1.524
     - 0.876
   * - 600
     - 2.696
     - 2.279
     - **1.980**
     - 3.589
     - 2.014
   * - 800
     - 5.017
     - 4.337
     - **3.784**
     - 6.560
     - 3.785
   * - 1000
     - >8s
     - 6.672
     - **5.967**
     - >8s
     - 5.400
   * - 1200
     - >8s
     - >8s
     - >8s
     - >8s
     - >8s

Using ``gurobipy``'s own docstring-idiomatic loop, it beats python-mip/Gurobi
on CPython for this benchmark (10–40% at n=200–600, both time out above
n=600) — profiling shows the difference is dominated by building large
*dense* Python-level sums (the objective alone sums :math:`n^2` terms), a
pattern closer to RCPSP's dense resource rows than to n-Queens' sparse
diagonal constraints. But the fair ``highspy`` comparison changes the
overall picture substantially: **``highspy``'s vectorized hl API is now the
fastest non-batch backend of all**, ahead of ``gurobipy``, python-mip/Gurobi,
and every other incremental-API option, at every size — and it's the only
non-batch backend that still finishes n=800–1000 within the 8 s budget on
CPython. This is the numpy vectorization paying off precisely where it
should: a genuinely dense, rectangular assignment structure with
elementwise arithmetic (``x * d[None, :]``) that maps directly onto numpy's
strengths.

Where python-mip still clearly wins for CFLP is **PyPy**: python-mip/Gurobi
on PyPy is roughly 2–3× faster than on CPython (e.g. 6.6s → 2.0s at n=600),
and since neither ``gurobipy`` (no PyPy wheel at all) nor ``highspy``'s
numpy-heavy APIs fare well on PyPy (the vectorized hl API is slower on PyPy
than on CPython at every size, the same pattern seen for n-Queens and TSP),
python-mip/CBC, /HiGHS and /Gurobi are the only backends that keep scaling
well under PyPy for this problem.

The ``highspy`` batch numpy API is still the fastest option in absolute
terms at large :math:`n` (1.7 s at n=1200 while every incremental-API
backend times out) — but building the CFLP CSR matrix by hand (``_cflp_csr``
in ``benchmarks/cflp_bench.py``) is roughly 100 lines of manual row/column
bookkeeping, versus the ~15-line natural formulation used for python-mip and
``gurobipy``, and about 10 lines for the vectorized ``highspy`` hl API. The
raw speed of the batch API comes at a real, measurable ergonomics cost that
the other three APIs avoid.

Run: ``python benchmarks/cflp_bench.py --build-only``


.. _bench-appendix-clarity:

Appendix: code clarity, incremental vs. vectorized modelling
--------------------------------------------------------------

The tables above establish the performance picture; this appendix backs up
the clarity claim with actual code. The short version: **python-mip's
incremental, ``model += ...`` style is consistently the most direct way to
express these models — it maps straight onto the mathematical formulation
regardless of whether the underlying structure is dense or sparse. The
vectorized ``highspy`` hl API can match that clarity, and its speed, but
only for problems that are naturally a dense rectangular array; forcing a
sparse problem into it costs both clarity and performance.**

n-Queens (dense :math:`n \times n` grid) — both styles read naturally:

.. code-block:: python

   # python-mip (n-Queens diagonal constraints)
   x = [[m.add_var(var_type=BINARY) for j in range(n)] for i in range(n)]
   for k in range(-(n - 1), n):
       cells = [x[i][i + k] for i in range(n) if 0 <= i + k < n]
       if len(cells) > 1:
           m += xsum(cells) <= 1

   # highspy vectorized hl API — same constraint
   x = h.addBinaries(n, n)
   for k in range(-(n - 1), n):
       diag = x.diagonal(k)
       if len(diag) > 1:
           h.addConstr(diag.sum() <= 1)

TSP single-commodity flow (sparse index set, arcs :math:`i \neq j` only) —
python-mip models only the arcs that exist; ``highspy``'s array API must
allocate the full dense matrix, including unused diagonal entries, and fix
them afterwards:

.. code-block:: python

   # python-mip — only real arcs (i != j) exist at all
   x = {(i, j): m.add_var(var_type=BINARY)
        for i in range(n) for j in range(n) if i != j}
   for i in range(n):
       m += xsum(x[i, j] for j in range(n) if j != i) == 1

   # highspy vectorized hl API — must allocate the full dense n x n
   # array (including the unused diagonal) to use axis-sum constraints,
   # then explicitly disable the diagonal entries afterwards
   x = h.addBinaries(n, n)
   for i in range(n):
       h.changeColBounds(x[i, i].index, 0, 0)   # arcs i == j don't exist
   h.addConstrs(x.sum(axis=1) == 1)

This is the same pattern reflected in the timing tables above: on dense,
rectangular problems (n-Queens, CFLP) the vectorized ``highspy`` API is
about as readable as python-mip *and* competitive on speed, occasionally
even the fastest incremental-style backend. On sparse/irregular problems
(TSP) it needs the dummy-diagonal workaround shown above, which is both
less direct to write and, once :math:`n` grows, slower than python-mip's
direct sparse formulation — because it pays :math:`O(n^2)` dense-array cost
for a problem that is genuinely :math:`O(n^2 - n)` in useful content only.

The ``highspy`` batch numpy API (hand-built CSR arrays) is a different
category entirely: it is the fastest option in every single benchmark in
this document, but it is not "incremental" or "vectorized-but-natural" — it
requires writing out row starts, column indices, coefficients and bounds by
hand as raw numpy arrays, with no per-variable/per-constraint objects
returned, no ``model +=`` style API, and no ability to add, inspect or
modify parts of the model incrementally afterwards. The ~100-line manual CSR
construction for CFLP's ``_cflp_csr`` function (versus ~15 lines for the
natural python-mip/``gurobipy`` formulation, and ~10 for the vectorized
``highspy`` hl API) is a fair representation of that gap in practice.


