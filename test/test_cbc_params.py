"""
Tests for CBC solver parameter correctness.

Covers: verbose (log suppression), threads, cut_passes, and preprocess.
Each test verifies that the parameter takes effect correctly and that
solver results remain correct (same optimal objective).
"""
import os
import multiprocessing
import pytest
import mip
from mip import CBC, Model, OptimizationStatus

INST_DIR = os.path.expanduser("~/inst/miplib/2017+spp/")

# air03: 124 rows, 10757 cols, set-covering — solves to optimality in ~1 s
AIR03 = os.path.join(INST_DIR, "air03.mps.gz")
AIR03_OPT = 340160.0  # known optimal (MIPLIB 2017)

TOL = 1.0  # absolute tolerance for objective comparison (integer problem)


def _needs_air03():
    if not os.path.exists(AIR03):
        pytest.skip(f"Instance not found: {AIR03}")


# ---------------------------------------------------------------------------
# verbose
# ---------------------------------------------------------------------------


def test_verbose_zero_suppresses_output(capfd):
    """verbose=0 must produce no solver output during read() and optimize()."""
    _needs_air03()
    m = Model(solver_name=CBC)
    m.verbose = 0
    m.read(AIR03)
    m.optimize()
    out, err = capfd.readouterr()
    assert out == "", f"Expected no stdout with verbose=0, got:\n{out[:400]}"
    assert err == "", f"Expected no stderr with verbose=0, got:\n{err[:400]}"


def test_verbose_one_produces_output(capfd):
    """verbose=1 must produce solver output during optimize()."""
    _needs_air03()
    m = Model(solver_name=CBC)
    m.verbose = 1
    m.read(AIR03)
    m.optimize()
    out, err = capfd.readouterr()
    assert (out + err) != "", "Expected some solver output with verbose=1"


def test_verbose_survives_read(capfd):
    """verbose=0 set before read() must survive the clear()+recreate cycle."""
    _needs_air03()
    m = Model(solver_name=CBC)
    m.verbose = 0
    # read() calls clear() internally, which recreates the solver
    m.read(AIR03)
    out, err = capfd.readouterr()
    assert out == "", f"verbose=0 lost after clear(): stdout was:\n{out[:400]}"
    assert err == "", f"verbose=0 lost after clear(): stderr was:\n{err[:400]}"


# ---------------------------------------------------------------------------
# threads
# ---------------------------------------------------------------------------


def test_threads_single_correct():
    """threads=1 (explicit) should give the same optimal as the default."""
    _needs_air03()
    m = Model(solver_name=CBC)
    m.verbose = 0
    m.read(AIR03)
    m.threads = 1
    status = m.optimize()
    assert status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - AIR03_OPT) <= TOL


@pytest.mark.skipif(
    multiprocessing.cpu_count() < 2, reason="Need at least 2 CPUs for threads test"
)
def test_threads_multi_correct():
    """threads=2 must produce the same optimal objective as single-thread."""
    _needs_air03()
    m = Model(solver_name=CBC)
    m.verbose = 0
    m.read(AIR03)
    m.threads = 2
    status = m.optimize()
    assert status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - AIR03_OPT) <= TOL


# ---------------------------------------------------------------------------
# cut_passes
# ---------------------------------------------------------------------------


def test_cut_passes_default_correct():
    """Default cut_passes (-1, let CBC decide) must reach optimality."""
    _needs_air03()
    m = Model(solver_name=CBC)
    m.verbose = 0
    m.read(AIR03)
    assert m.cut_passes == -1
    status = m.optimize()
    assert status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - AIR03_OPT) <= TOL


def test_cut_passes_explicit_correct():
    """Explicit cut_passes=5 (user override) must still reach optimality."""
    _needs_air03()
    m = Model(solver_name=CBC)
    m.verbose = 0
    m.read(AIR03)
    m.cut_passes = 5
    status = m.optimize()
    assert status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - AIR03_OPT) <= TOL


def test_cut_passes_zero():
    """cut_passes=0 (no cuts) must still reach optimality for air03."""
    _needs_air03()
    m = Model(solver_name=CBC)
    m.verbose = 0
    m.read(AIR03)
    m.cut_passes = 0
    status = m.optimize()
    assert status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - AIR03_OPT) <= TOL


# ---------------------------------------------------------------------------
# preprocess (CglPreProcess — MIP preprocessing / SOS detection)
# ---------------------------------------------------------------------------


def test_preprocess_off_correct():
    """preprocess=0 (CglPreProcess disabled) must still reach optimality."""
    _needs_air03()
    m = Model(solver_name=CBC)
    m.verbose = 0
    m.read(AIR03)
    m.preprocess = 0
    status = m.optimize()
    assert status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - AIR03_OPT) <= TOL


def test_preprocess_default_correct():
    """Default preprocess (CglPreProcess enabled) must reach optimality."""
    _needs_air03()
    m = Model(solver_name=CBC)
    m.verbose = 0
    m.read(AIR03)
    assert m.preprocess == -1  # default: let CBC decide
    status = m.optimize()
    assert status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - AIR03_OPT) <= TOL


def test_preprocess_sos_correct():
    """preprocess=1 (SOS preprocessing) must still reach optimality."""
    _needs_air03()
    m = Model(solver_name=CBC)
    m.verbose = 0
    m.read(AIR03)
    m.preprocess = 1
    status = m.optimize()
    assert status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - AIR03_OPT) <= TOL


# ---------------------------------------------------------------------------
# LP time limit, LP method, LP iteration limit
# ---------------------------------------------------------------------------

import math
import time
from mip import LP_Method

# brazil3: 14646 rows, 23968 cols — LP takes ~16 s, ideal for limit tests
BRAZIL3 = os.path.join(INST_DIR, "brazil3.mps.gz")

# sp150x300d: smaller LP that reaches optimal, used for LP method tests
SP150 = os.path.join(INST_DIR, "sp150x300d.mps.gz")


def _needs_brazil3():
    if not os.path.exists(BRAZIL3):
        pytest.skip(f"Instance not found: {BRAZIL3}")


def _needs_sp150():
    if not os.path.exists(SP150):
        pytest.skip(f"Instance not found: {SP150}")


def test_lp_time_limit_truncated():
    """LP solve stopped by time limit must return TRUNCATED, not ERROR."""
    _needs_brazil3()
    m = Model(solver_name=CBC)
    m.verbose = 0
    m.read(BRAZIL3)
    t0 = time.time()
    status = m.optimize(relax=True, max_seconds=3)
    elapsed = time.time() - t0
    assert status == OptimizationStatus.TRUNCATED, f"Expected TRUNCATED, got {status}"
    assert elapsed < 10, f"Solver ran too long: {elapsed:.1f}s (expected ≤10s)"
    # A dual bound should be available (dual simplex is default for LP)
    assert math.isfinite(m.objective_bound), "Expected a finite dual bound"


def test_lp_iter_limit_truncated():
    """LP solve stopped by iteration limit must return TRUNCATED."""
    _needs_brazil3()
    m = Model(solver_name=CBC)
    m.verbose = 0
    m.read(BRAZIL3)
    m.max_iter = 500
    t0 = time.time()
    status = m.optimize(relax=True)
    elapsed = time.time() - t0
    assert status == OptimizationStatus.TRUNCATED, f"Expected TRUNCATED, got {status}"
    # Should stop quickly (well under 16 s)
    assert elapsed < 10, f"Took too long: {elapsed:.1f}s (expected to stop at 500 iter)"


def test_lp_method_dual():
    """Dual simplex must reach optimality on sp150x300d LP."""
    _needs_sp150()
    m = Model(solver_name=CBC)
    m.verbose = 0
    m.lp_method = LP_Method.DUAL
    m.read(SP150)
    status = m.optimize(relax=True)
    assert status == OptimizationStatus.OPTIMAL, f"Expected OPTIMAL, got {status}"


def test_lp_method_primal():
    """Primal simplex must reach optimality on sp150x300d LP."""
    _needs_sp150()
    m = Model(solver_name=CBC)
    m.verbose = 0
    m.lp_method = LP_Method.PRIMAL
    m.read(SP150)
    status = m.optimize(relax=True)
    assert status == OptimizationStatus.OPTIMAL, f"Expected OPTIMAL, got {status}"


def test_lp_method_barrier():
    """Barrier must reach optimality on sp150x300d LP."""
    _needs_sp150()
    m = Model(solver_name=CBC)
    m.verbose = 0
    m.lp_method = LP_Method.BARRIER
    m.read(SP150)
    status = m.optimize(relax=True)
    assert status == OptimizationStatus.OPTIMAL, f"Expected OPTIMAL, got {status}"
