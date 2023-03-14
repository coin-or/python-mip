"""Set of tests for solving the LP relaxation"""

from glob import glob
from os import environ
import json
from itertools import product
import pytest
from mip import CBC, GUROBI, OptimizationStatus
from mip_rcpsp import create_mip

INSTS = glob("./data/rcpsp*.json") + glob("./test/data/rcpsp*.json")

TOL = 1e-4

SOLVERS = [CBC]
if "GUROBI_HOME" in environ:
    SOLVERS += [GUROBI]


@pytest.mark.parametrize("solver, instance", product(SOLVERS, INSTS))
def test_rcpsp_relax(solver: str, instance: str):
    """tests the solution of the LP relaxation of different rcpsp instances"""
    with open(instance, "r") as finst:
        data = json.load(finst)
        J = data["J"]
        d = data["d"]
        S = data["S"]
        c = data["c"]
        r = data["r"]
        EST = data["EST"]
        z_relax = data["z_relax"]
    # print("test %s %s" % (solver, instance))
    mip = create_mip(solver, J, d, S, c, r, EST, True)
    mip.verbose = 0
    mip.optimize()
    assert mip.status == OptimizationStatus.OPTIMAL
    assert abs(z_relax - mip.objective_value) <= TOL


@pytest.mark.parametrize("solver, instance", product(SOLVERS, INSTS))
def test_rcpsp_relax_mip(solver: str, instance: str):
    """tests the solution of the LP relaxation of different rcpsp instances"""
    with open(instance, "r") as finst:
        data = json.load(finst)
        J = data["J"]
        d = data["d"]
        S = data["S"]
        c = data["c"]
        r = data["r"]
        EST = data["EST"]
        z_relax = data["z_relax"]
    # print("test %s %s" % (solver, instance))
    mip = create_mip(solver, J, d, S, c, r, EST, False)
    mip.verbose = 0
    mip.relax()
    mip.optimize()
    assert mip.status == OptimizationStatus.OPTIMAL
    assert abs(z_relax - mip.objective_value) <= TOL


@pytest.mark.parametrize("solver, instance", product(SOLVERS, INSTS))
def test_rcpsp_mip(solver: str, instance: str):
    """tests the solution of different RCPSP MIPs"""
    with open(instance, "r") as finst:
        data = json.load(finst)
        J = data["J"]
        d = data["d"]
        S = data["S"]
        c = data["c"]
        r = data["r"]
        EST = data["EST"]
        z_relax = data["z_relax"]
        z_ub = data["z_ub"]
    # print("test %s %s" % (solver, instance))
    mip = create_mip(solver, J, d, S, c, r, EST, False)
    mip.verbose = 0
    mip.optimize(max_nodes=64)
    assert mip.status not in [
        OptimizationStatus.INFEASIBLE,
        OptimizationStatus.INT_INFEASIBLE,
        OptimizationStatus.UNBOUNDED,
        OptimizationStatus.ERROR,
        OptimizationStatus.CUTOFF,
        OptimizationStatus.LOADED,
        OptimizationStatus.OTHER,
    ]
    assert z_relax - TOL <= mip.objective_bound <= z_ub + TOL
    if mip.status in [OptimizationStatus.OPTIMAL]:
        assert abs(mip.objective_value - mip.objective_bound) <= TOL
    if mip.status in [
        OptimizationStatus.OPTIMAL,
        OptimizationStatus.FEASIBLE,
    ]:
        assert z_relax - TOL <= mip.objective_value
        tl = int(round(mip.objective_value))
        assert mip.vars["x(%d,%d)" % (J[-1], tl)].x >= 0.99
        xOn = [v for v in mip.vars if v.x >= 0.99 and v.name.startswith("x(")]
        assert len(xOn) == len(J)

@pytest.mark.parametrize("solver, instance", product(SOLVERS, INSTS))
def test_rcpsp_mipstart(solver: str, instance: str):
    """tests the solution of different rcpsps MIPs with uwing MIPStarts"""
    with open(instance, "r") as finst:
        data = json.load(finst)
        J = data["J"]
        d = data["d"]
        S = data["S"]
        c = data["c"]
        r = data["r"]
        EST = data["EST"]
        z_ub = data["z_ub"]
        mipstart = data["mipstart"]
    # print("test %s %s" % (solver, instance))
    mip = create_mip(solver, J, d, S, c, r, EST, False)
    mip.verbose = 0
    mip.pump_passes = 0
    mip.cuts = 0
    mip.cut_passes = 0
    mip.start = [(mip.var_by_name(n), v) for (n, v) in mipstart]
    mip.optimize(max_nodes=3)
    assert mip.status in [OptimizationStatus.FEASIBLE, 
                          OptimizationStatus.OPTIMAL]
    assert abs(mip.objective_value - z_ub) <= TOL
