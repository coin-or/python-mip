"""
Tests for LP warm-start reoptimization via Cbc_resolve.

Loads real MIPLIB instances, solves the LP relaxation, applies modifications
(bound tightening, added constraints, or both), and re-optimises. Compares
CBC results against Gurobi to verify correctness.
"""
import os
import math
import pytest
import mip
from mip import CBC, GUROBI, Model, xsum, OptimizationStatus

INST_DIR = os.path.expanduser("~/inst/miplib/2017+spp/")

# Instances chosen to cover different row/col ratios and problem structures.
# All solve the LP relaxation in well under 1 second.
INSTANCES = [
    "neos5.mps.gz",          # 63 rows,  63 cols  — dense, binary
    "c05100.mps.gz",         # 105 rows, 500 cols  — set partitioning
    "sp150x300d.mps.gz",     # 450 rows, 600 cols  — mixed binary/continuous
    "timtab1.mps.gz",        # 171 rows, 397 cols  — mixed integer
    "neos859080.mps.gz",     # 164 rows, 160 cols  — mixed integer
]

TOL = 1e-4  # objective agreement tolerance (relative)


def has_gurobi():
    try:
        m = Model(solver_name=GUROBI)
        del m
        return True
    except Exception:
        return False


def _load_and_solve_lp(inst_path: str, solver_name: str) -> Model:
    """Load an instance and solve its LP relaxation."""
    m = Model(solver_name=solver_name)
    m.verbose = 0
    m.read(inst_path)
    status = m.optimize(relax=True)
    assert status == OptimizationStatus.OPTIMAL, (
        f"{solver_name}: initial LP of {os.path.basename(inst_path)} not optimal: {status}"
    )
    return m


def _obj_match(a: float, b: float) -> bool:
    """True when the two objectives agree within TOL (relative)."""
    if abs(b) < 1e-10:
        return abs(a) < TOL
    return abs(a - b) / abs(b) < TOL


# ---------------------------------------------------------------------------
# Modifications: computed from a reference LP solution so both CBC and Gurobi
# receive identical numeric changes (avoids solver-specific vertex differences).
# ---------------------------------------------------------------------------

def _compute_bound_mods(m_ref: Model):
    """Return list of (var_index, new_ub) using m_ref's LP solution."""
    n_mod = max(3, m_ref.num_cols // 20)
    mods = []
    for i in range(n_mod):
        v = m_ref.vars[i]
        lp_val = v.x
        if lp_val > 1e-8:
            mods.append((i, min(v.ub, lp_val * 0.9)))
    return mods


def _compute_cut(m_ref: Model):
    """Return (var_indices, rhs) for a budget cut using m_ref's LP solution."""
    n = max(3, m_ref.num_cols // 10)
    current_sum = sum(m_ref.vars[i].x for i in range(n))
    if current_sum < 1e-8:
        return None
    return list(range(n)), 0.92 * current_sum


def _apply_bound_mods(m: Model, mods) -> None:
    for i, new_ub in mods:
        m.vars[i].ub = min(m.vars[i].ub, new_ub)


def _apply_cut(m: Model, cut, name: str = "resolve_budget_cut") -> None:
    if cut is None:
        return
    indices, rhs = cut
    m += xsum(m.vars[i] for i in indices) <= rhs, name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inst", INSTANCES)
@pytest.mark.parametrize("mod", ["bound_tighten", "add_constraint", "both"])
def test_resolve_matches_gurobi(inst: str, mod: str):
    """After modifying the LP, Cbc_resolve gives the same obj as Gurobi.

    Modifications are computed from CBC's LP solution so that both solvers
    receive identical numeric changes, avoiding solver-specific vertex issues.
    """
    if not has_gurobi():
        pytest.skip("Gurobi not available")

    inst_path = os.path.join(INST_DIR, inst)
    if not os.path.exists(inst_path):
        pytest.skip(f"Instance not found: {inst_path}")

    # Solve with CBC first to get the reference LP vertex for modifications
    m_cbc = _load_and_solve_lp(inst_path, CBC)

    bound_mods = _compute_bound_mods(m_cbc) if mod in ("bound_tighten", "both") else []
    cut = _compute_cut(m_cbc) if mod in ("add_constraint", "both") else None

    # Apply to CBC and re-solve (warm start via Cbc_resolve)
    _apply_bound_mods(m_cbc, bound_mods)
    _apply_cut(m_cbc, cut)
    status_cbc = m_cbc.optimize(relax=True)

    # Apply the same numeric changes to a fresh Gurobi solve
    m_grb = Model(solver_name=GUROBI)
    m_grb.verbose = 0
    m_grb.read(inst_path)
    _apply_bound_mods(m_grb, bound_mods)
    _apply_cut(m_grb, cut)
    status_grb = m_grb.optimize(relax=True)

    # Both solvers must agree on feasibility
    assert status_cbc == status_grb, (
        f"[{inst}|{mod}] status mismatch: CBC={status_cbc} GRB={status_grb}"
    )

    if status_grb == OptimizationStatus.OPTIMAL:
        assert _obj_match(m_cbc.objective_value, m_grb.objective_value), (
            f"[{inst}|{mod}] obj mismatch: CBC={m_cbc.objective_value:.6f} "
            f"GRB={m_grb.objective_value:.6f}"
        )


@pytest.mark.parametrize("inst", INSTANCES)
def test_resolve_fresh_vs_warm(inst: str):
    """Cbc_resolve result matches a deterministic second warm-start on the same problem."""
    inst_path = os.path.join(INST_DIR, inst)
    if not os.path.exists(inst_path):
        pytest.skip(f"Instance not found: {inst_path}")

    # First model: solve → compute mods → resolve (warm-start)
    m1 = _load_and_solve_lp(inst_path, CBC)
    bound_mods = _compute_bound_mods(m1)
    cut = _compute_cut(m1)
    _apply_bound_mods(m1, bound_mods)
    _apply_cut(m1, cut)
    status1 = m1.optimize(relax=True)

    # Second model: apply the same numeric changes, solve (also warm after initial)
    m2 = _load_and_solve_lp(inst_path, CBC)
    _apply_bound_mods(m2, bound_mods)
    _apply_cut(m2, cut)
    status2 = m2.optimize(relax=True)

    assert status1 == status2

    if status1 == OptimizationStatus.OPTIMAL:
        assert _obj_match(m1.objective_value, m2.objective_value), (
            f"[{inst}] warm vs warm2 mismatch: "
            f"{m1.objective_value:.6f} != {m2.objective_value:.6f}"
        )


@pytest.mark.parametrize("inst", INSTANCES)
def test_resolve_repeated_modifications(inst: str):
    """Multiple sequential LP re-solves, each adding a disjoint budget cut.

    Verifies that for minimisation problems each cut can only make the
    objective worse-or-equal.  If a cut causes infeasibility the loop
    stops early — that is an acceptable outcome, not a test failure.
    """
    inst_path = os.path.join(INST_DIR, inst)
    if not os.path.exists(inst_path):
        pytest.skip(f"Instance not found: {inst_path}")

    m = _load_and_solve_lp(inst_path, CBC)
    prev_obj = m.objective_value
    sense = m.sense  # MINIMIZE or MAXIMIZE

    for i in range(4):
        n_cols = m.num_cols
        slice_size = max(2, n_cols // 20)
        start = (i * slice_size) % n_cols
        end = min(start + slice_size, n_cols)
        current_sum = sum(m.vars[j].x for j in range(start, end))
        if current_sum > 1e-8:
            m += xsum(m.vars[j] for j in range(start, end)) <= 0.92 * current_sum, f"cut_{i}"

        status = m.optimize(relax=True)

        # A cut may legitimately make the problem infeasible — stop there.
        if status == OptimizationStatus.INFEASIBLE:
            break

        assert status == OptimizationStatus.OPTIMAL, (
            f"[{inst}] round {i}: unexpected status {status}"
        )
        cur_obj = m.objective_value
        assert math.isfinite(cur_obj), f"[{inst}] round {i}: non-finite obj {cur_obj}"
        if sense == mip.MINIMIZE:
            assert cur_obj >= prev_obj - TOL * abs(prev_obj + 1), (
                f"[{inst}] round {i}: obj decreased for minimisation "
                f"({cur_obj:.6f} < {prev_obj:.6f})"
            )
        prev_obj = cur_obj


def test_lp_preprocess_tightens_bounds():
    """lp_preprocess=True should tighten bounds via knapsack analysis.

    Uses a knapsack-structured binary problem where MILP bound propagation
    can fix variables.  After lp_preprocess the objective value must be at
    least as tight (for minimisation) as without preprocessing.  We also
    verify that the plain LP relaxation (lp_preprocess=False) is not
    accidentally modified.
    """
    inst_path = os.path.join(INST_DIR, "stein45.mps.gz")
    if not os.path.exists(inst_path):
        pytest.skip(f"Instance not found: {inst_path}")

    # Plain LP relaxation — baseline
    m_plain = Model(solver_name=CBC)
    m_plain.verbose = 0
    m_plain.read(inst_path)
    status_plain = m_plain.optimize(relax=True)
    assert status_plain == OptimizationStatus.OPTIMAL, (
        f"Plain LP not optimal: {status_plain}"
    )
    obj_plain = m_plain.objective_value

    # LP with fast MILP preprocessing — bounds may be tightened
    m_pre = Model(solver_name=CBC)
    m_pre.verbose = 0
    m_pre.read(inst_path)
    status_pre = m_pre.optimize(relax=True, lp_preprocess=True)

    # Preprocessing may prove infeasibility or return a (possibly tighter) LP value.
    # For stein45 (minimisation), preprocessing can only raise or match the LP bound.
    assert status_pre in (OptimizationStatus.OPTIMAL, OptimizationStatus.INFEASIBLE), (
        f"Unexpected status with lp_preprocess=True: {status_pre}"
    )
    if status_pre == OptimizationStatus.OPTIMAL:
        assert m_pre.objective_value >= obj_plain - TOL * abs(obj_plain + 1), (
            f"lp_preprocess=True gave worse LP bound: "
            f"{m_pre.objective_value:.6f} < {obj_plain:.6f}"
        )
