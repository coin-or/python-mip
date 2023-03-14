"""Set of tests for solving the LP relaxation"""

from glob import glob
from os import environ
import json
from itertools import product
import pytest
from mip import CBC, GUROBI, OptimizationStatus
from mip_2d_pack import create_mip

INSTS = glob("./data/two_dim_pack_p*.json") + glob(
    "./test/data/two_dim_pack_*.json"
)

TOL = 1e-4

SOLVERS = [CBC]
if "GUROBI_HOME" in environ:
    SOLVERS += [GUROBI]


@pytest.mark.parametrize("solver, instance", product(SOLVERS, INSTS))
def test_2dpack_relax_and_cut(solver: str, instance: str):
    """tests the solution of the LP relaxation of different 2D pack instances"""
    with open(instance, "r") as finst:
        data = json.load(finst)
        W = data["W"]
        w = data["w"]
        h = data["h"]
        obj_relax = data["relax"]
        best = data["best"]

    # print("test %s %s" % (solver, instance))
    mip = create_mip(solver, w, h, W, False)
    mip.verbose = 0
    mip.optimize(relax=True)
    assert mip.status == OptimizationStatus.OPTIMAL
    assert abs(obj_relax - mip.objective_value) <= TOL

    cuts = mip.generate_cuts()

    if cuts:
        mip += cuts
        mip.optimize(relax=True)

    sobj = mip.objective_value

    assert sobj <= best + 1e-5


@pytest.mark.parametrize("solver, instance", product(SOLVERS, INSTS))
def test_2dpack_mip(solver: str, instance: str):
    """tests the MIP solution of different 2D pack instances"""
    with open(instance, "r") as finst:
        data = json.load(finst)
        W = data["W"]
        w = data["w"]
        h = data["h"]
        z_relax = data["relax"]
        z_ub = data["best"]
        isopt = data["opt"]

    mip = create_mip(solver, w, h, W, False)
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

        if isopt:
            if mip.status == OptimizationStatus.OPTIMAL:
                assert abs(z_ub - mip.objective_value) <= TOL
            else:
                assert mip.objective_value >= z_ub - TOL

        sumh = 0
        for v in [
            v
            for v in mip.vars
            if v.x >= 0.99 and v.x <= 1.01 and v.name.startswith("x(")
        ]:
            i = int(v.name.split(",")[0].split("(")[1])
            j = int(v.name.split(",")[1].split(")")[0])
            if i == j:
                sumh += v.obj

        assert (sumh - mip.objective_value) <= TOL

        na = sum(
            [
                1
                for v in mip.vars
                if v.x >= 0.99 and v.x <= 1.01 and v.name.startswith("x(")
            ]
        )
        assert na == len(w)
