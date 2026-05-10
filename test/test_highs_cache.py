"""Robustness tests for the HiGHS column/row cache.

Exercises every flush trigger in isolation and in combination, verifying that
results are numerically identical regardless of build order or mid-build query.
"""

import os
import tempfile

import pytest

import mip
from mip import HIGHS, Model, OptimizationStatus, xsum
from mip.highs import _CACHE_INITIAL_CAP

pytestmark = pytest.mark.skipif(
    not mip.highs.has_highs, reason="HiGHS not available"
)


def h(**kwargs) -> Model:
    """Convenience: silent HiGHS model."""
    m = Model(solver_name=HIGHS, **kwargs)
    m.verbose = 0
    return m


# ── helpers ───────────────────────────────────────────────────────────────────

def _committed(m: Model):
    s = m.solver
    return s._col_committed, s._row_committed


def _pending(m: Model):
    s = m.solver
    return s._col_fill, s._row_fill


# ─────────────────────────────────────────────────────────────────────────────
# 1. num_cols / num_rows reflect committed + pending without flushing
# ─────────────────────────────────────────────────────────────────────────────

def test_virtual_counts():
    m = h()
    assert m.solver.num_cols() == 0
    assert m.solver.num_rows() == 0

    x = m.add_var()
    assert m.solver.num_cols() == 1
    assert _committed(m) == (0, 0)   # nothing flushed yet

    y = m.add_var()
    assert m.solver.num_cols() == 2

    m += x + y <= 10
    assert m.solver.num_rows() == 1
    assert _committed(m) == (0, 0)   # still nothing flushed

    m += x + y >= 5
    assert m.solver.num_rows() == 2
    assert _pending(m) == (2, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Standard cols-then-rows build (baseline)
# ─────────────────────────────────────────────────────────────────────────────

def test_cols_then_rows():
    m = h(sense=mip.MAXIMIZE)
    x = m.add_var(ub=10)
    y = m.add_var(ub=5)
    z = m.add_var(ub=3)
    m += x + y <= 8
    m += x + z <= 6
    m.objective = x + 2 * y + z
    assert m.optimize() == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - 16.0) < 1e-6
    assert abs(x.x - 3.0) < 1e-6
    assert abs(y.x - 5.0) < 1e-6
    assert abs(z.x - 3.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 3. Interleaved: cols / rows / cols / rows
# ─────────────────────────────────────────────────────────────────────────────

def test_interleaved_build():
    """Add vars, add constrs, add more vars, add more constrs — no explicit flush."""
    m = h(sense=mip.MAXIMIZE)
    x = m.add_var(ub=10)
    y = m.add_var(ub=5)
    m += x + y <= 8          # row cached; refs col indices 0, 1
    z = m.add_var(ub=3)      # col index 2 (committed=0, fill=2 before this)
    m += x + z <= 6          # row cached; refs col indices 0, 2
    m.objective = x + 2 * y + z
    assert m.optimize() == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - 16.0) < 1e-6


def test_many_interleave_batches():
    """Several add-var / add-constr alternations; each row references the most
    recently added variable to stress index tracking."""
    m = h(sense=mip.MAXIMIZE)
    n = 10
    xs = []
    for i in range(n):
        v = m.add_var(ub=1.0)
        xs.append(v)
        if i > 0:
            # x[i-1] + x[i] <= 1  (adjacent-pair constraint)
            m += xs[i - 1] + xs[i] <= 1.0
    m.objective = xsum(xs)
    # Independent set on a path graph: ceil(n/2) = 5
    assert m.optimize() == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - 5.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 4. Flush triggered by reading a variable property mid-build
# ─────────────────────────────────────────────────────────────────────────────

def test_flush_on_var_get_lb():
    m = h(sense=mip.MAXIMIZE)
    x = m.add_var(lb=2.0, ub=10.0)
    y = m.add_var(lb=0.0, ub=5.0)
    # Reading lb triggers _flush(); x,y are now committed.
    assert abs(x.lb - 2.0) < 1e-9
    assert _committed(m) == (2, 0)
    assert _pending(m) == (0, 0)

    z = m.add_var(ub=3.0)      # new pending col after flush
    m += x + y + z <= 12.0
    m.objective = x + y + z
    assert m.optimize() == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - 12.0) < 1e-6


def test_flush_on_var_get_ub():
    m = h()
    x = m.add_var(ub=7.0)
    assert abs(x.ub - 7.0) < 1e-9
    assert _committed(m) == (1, 0)


def test_flush_on_var_set_ub():
    """Modify a bound after the var was cached but not yet committed."""
    m = h(sense=mip.MAXIMIZE)
    x = m.add_var(ub=5.0)
    y = m.add_var(ub=5.0)
    # Setting ub flushes, then modifies in HiGHS.
    x.ub = 3.0
    m += x + y <= 6.0
    m.objective = x + y
    assert m.optimize() == OptimizationStatus.OPTIMAL
    # maximize x+y, x+y<=6, x<=3, y<=5 → obj=6 (e.g. x=1,y=5 or x=3,y=3)
    assert abs(m.objective_value - 6.0) < 1e-6
    assert x.x <= 3.0 + 1e-6   # ub respected


def test_flush_on_constr_get_rhs():
    m = h()
    x = m.add_var(ub=10)
    y = m.add_var(ub=10)
    c = m.add_constr(x + y <= 8, "c1")
    assert abs(c.rhs - 8.0) < 1e-9  # triggers flush
    assert _committed(m) == (2, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Flush triggered by writing a model file mid-build
# ─────────────────────────────────────────────────────────────────────────────

def test_flush_on_write_lp():
    m = h(sense=mip.MAXIMIZE)
    x = m.add_var(name="x", ub=10.0)
    y = m.add_var(name="y", ub=5.0)
    m += x + y <= 8.0, "cap"
    with tempfile.NamedTemporaryFile(suffix=".lp", delete=False) as f:
        path = f.name
    try:
        m.write(path)
        assert _committed(m) == (2, 1)
        # Continue building and solving after the write
        z = m.add_var(name="z", ub=3.0)
        m += x + z <= 6.0
        m.objective = x + 2 * y + z
        assert m.optimize() == OptimizationStatus.OPTIMAL
        assert abs(m.objective_value - 16.0) < 1e-6
    finally:
        os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Column parameter: new variable references an existing (committed) row
# ─────────────────────────────────────────────────────────────────────────────

def test_column_parameter():
    """add_var with column= flushes first, adds var via single-col path."""
    m = h(sense=mip.MAXIMIZE)
    x = m.add_var(ub=5.0)
    y = m.add_var(ub=5.0)
    # Commit x, y and one row.
    c1 = m.add_constr(x + y <= 8.0, "c1")
    m.solver._flush()   # explicit flush to pre-commit c1 so col= works
    assert _committed(m) == (2, 1)

    # Add z that appears in c1 with coefficient 1 (makes c1: x+y+z<=8).
    z = m.add_var(ub=3.0, column=mip.Column([c1], [1.0]))
    assert _committed(m) == (3, 1)   # column= path commits immediately

    m.objective = x + y + z
    assert m.optimize() == OptimizationStatus.OPTIMAL
    # Best: x+y+z<=8, x<=5,y<=5,z<=3 → x=2,y=3,z=3=8; obj=8
    assert abs(m.objective_value - 8.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 7. remove_vars mid-build: committed counts stay in sync
# ─────────────────────────────────────────────────────────────────────────────

def test_remove_vars_mid_build():
    m = h(sense=mip.MAXIMIZE)
    x = m.add_var(name="x", ub=10.0)
    y = m.add_var(name="y", ub=5.0)
    tmp = m.add_var(name="tmp", ub=100.0)   # will be removed
    m += x + y <= 8.0

    # Removing triggers flush + Highs_deleteColsBySet + committed count update.
    m.vars.remove([tmp])
    assert m.solver._col_committed == 2
    assert m.solver._col_fill == 0

    z = m.add_var(name="z", ub=3.0)        # gets committed index 2 after removal
    m += x + z <= 6.0
    m.objective = x + 2 * y + z
    assert m.optimize() == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - 16.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 8. remove_constrs mid-build: row committed count stays in sync
# ─────────────────────────────────────────────────────────────────────────────

def test_remove_constrs_mid_build():
    m = h(sense=mip.MAXIMIZE)
    x = m.add_var(ub=10.0)
    y = m.add_var(ub=5.0)
    tight = m.add_constr(x + y <= 3.0, "tight")   # will be removed
    m += x <= 8.0

    m.constrs.remove([tight])
    assert m.solver._row_committed == 1   # only x<=8 remains
    assert m.solver._row_fill == 0

    # Now add more rows
    m += y <= 4.0
    m.objective = x + y
    assert m.optimize() == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - 12.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 9. set_objective mid-build (triggers flush)
# ─────────────────────────────────────────────────────────────────────────────

def test_set_objective_mid_build():
    m = h(sense=mip.MAXIMIZE)
    x = m.add_var(ub=5.0)
    y = m.add_var(ub=5.0)
    # set_objective flushes x, y, then continues
    m.objective = x + 2 * y
    assert _committed(m) == (2, 0)

    z = m.add_var(ub=3.0)
    m += x + y + z <= 10.0
    # Set objective again including z — triggers flush of z
    m.objective = x + 2 * y + z
    assert m.optimize() == OptimizationStatus.OPTIMAL
    # x=2, y=5, z=3 → obj=15, x+y+z=10≤10 ✓
    assert abs(m.objective_value - 15.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 10. solve LP (relax=True path) with interleaved build
# ─────────────────────────────────────────────────────────────────────────────

def test_relax_with_interleaved_build():
    m = h(sense=mip.MAXIMIZE)
    x = m.add_var(var_type=mip.BINARY)
    y = m.add_var(var_type=mip.BINARY)
    m += x + y <= 1.0
    z = m.add_var(var_type=mip.BINARY)
    m += y + z <= 1.0
    m.objective = x + y + z
    # LP relaxation: x=1, y=0, z=1 or x=0.5, y=0.5, z=0.5
    status = m.solver.optimize(relax=True)
    assert status == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - 2.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 11. Grow beyond _CACHE_INITIAL_CAP (tests _grow_cols / _grow_row_nz)
# ─────────────────────────────────────────────────────────────────────────────

def test_grow_beyond_initial_cap():
    """Build a model larger than _CACHE_INITIAL_CAP without any forced flush."""
    n = _CACHE_INITIAL_CAP + 500   # guaranteed to trigger grow
    m = h(sense=mip.MAXIMIZE)
    xs = [m.add_var(ub=1.0) for _ in range(n)]
    # One giant constraint: sum(x) <= n/2
    m += xsum(xs) <= n / 2
    m.objective = xsum(xs)

    # No flush should have happened yet for cols (rows might trigger col flush)
    # (xsum constraint is 1 row with n NZs — _grow_row_nz fires here)
    assert m.optimize() == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - n / 2) < 1e-4


def test_grow_rows_beyond_initial_cap():
    """More rows than _CACHE_INITIAL_CAP."""
    n = _CACHE_INITIAL_CAP + 100
    m = h(sense=mip.MAXIMIZE)
    xs = [m.add_var(ub=1.0) for _ in range(n)]
    for v in xs:
        m += v <= 1.0        # n individual bounds (≥ cap)
    m.objective = xsum(xs)
    assert m.optimize() == OptimizationStatus.OPTIMAL
    assert abs(m.objective_value - n) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# 12. constr_get_expr (full round-trip) after interleaved build
# ─────────────────────────────────────────────────────────────────────────────

def test_constr_get_expr_after_interleaved():
    m = h()
    x = m.add_var(name="x", ub=10.0)
    y = m.add_var(name="y", ub=10.0)
    c = m.add_constr(2 * x + 3 * y <= 12.0, "c")
    z = m.add_var(name="z", ub=10.0)
    m += z <= 5.0

    # Accessing c.expr triggers flush of all pending cols/rows.
    expr = c.expr
    coeffs = {v.name: coef for v, coef in expr.expr.items()}
    assert abs(coeffs.get("x", 0) - 2.0) < 1e-9
    assert abs(coeffs.get("y", 0) - 3.0) < 1e-9
    # LinExpr stores constraint as lhs+const<=0, so const = -rhs
    assert abs(expr.const + 12.0) < 1e-9   # i.e. const == -12


# ─────────────────────────────────────────────────────────────────────────────
# 13. Named cols and rows survive flush correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_names_survive_flush():
    m = h()
    x = m.add_var(name="alice", ub=1.0)
    y = m.add_var(name="bob", ub=1.0)
    m += x + y <= 1.0, "cap"
    m.optimize()   # triggers flush

    assert m.vars["alice"].idx == 0
    assert m.vars["bob"].idx == 1
    assert m.constrs["cap"].idx == 0


# ─────────────────────────────────────────────────────────────────────────────
# 14. Integer model: interleaved binary vars and constraints
# ─────────────────────────────────────────────────────────────────────────────

def test_mip_interleaved():
    """Small MIP with interleaved add_var / add_constr."""
    m = h(sense=mip.MAXIMIZE)
    # Knapsack: items have weight and value
    weights = [2, 3, 4, 5]
    values  = [3, 4, 5, 6]
    cap = 8
    xs = []
    for w, v in zip(weights, values):
        xv = m.add_var(var_type=mip.BINARY)
        xs.append(xv)
        # Add a dummy constr between vars to stress interleaving
        if len(xs) > 1:
            m += xs[-2] + xs[-1] <= 1   # at most one of each pair
    m += xsum(w * x for w, x in zip(weights, xs)) <= cap
    m.objective = xsum(v * x for v, x in zip(values, xs))
    assert m.optimize() == OptimizationStatus.OPTIMAL
    assert m.objective_value > 0
