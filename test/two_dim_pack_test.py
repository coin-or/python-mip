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
def test_rcpsp_relax(solver: str, instance: str):
    """tests the solution of the LP relaxation of different rcpsp instances"""
    with open(instance, "r") as finst:
        data = json.load(finst)
        W = data["W"]
        w = data["w"]
        h = data["h"]
        obj_relax = data["relax"]

    # print("test %s %s" % (solver, instance))
    mip = create_mip(solver, w, h, W, True)
    mip.verbose = 0
    mip.optimize()
    assert mip.status == OptimizationStatus.OPTIMAL
    assert abs(obj_relax - mip.objective_value) <= TOL
