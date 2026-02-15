"""Test MIP optimization reading files in .mps and .lp,  compressed or not"""


from glob import glob
from itertools import product
from os import environ
from os.path import basename, dirname, join, exists

import pytest

import mip.gurobi
import mip.highs
from mip import Model, OptimizationStatus, CBC, GUROBI, HIGHS
from util import skip_on

# for each MIP in test/data, best lower and upper bounds
# to be used when checking optimization results
BOUNDS = {
    "bft": (-995, -995),
    "1443_0-9": (-590001.9, -590001.9),
    "2050_3_7": (-43751.7, -43751.7),
    "gesa3_o": (2.7991e07, 2.7991e07),
    "flugpl": (1.2015e06, 1.2015e06),
    "gesa2": (2.57797e07, 2.57822e07),
    "gesa3": (2.7991e07, 2.7991e07),
    "qiu": (-359.937, -128.467),
    "10teams": (924, 924),
    "air03": (340160, 340160),
    "air04": (56137, 56137),
    "air05": (26154, 26402),
    "arki001": (7.58028e06, 7.58248e06),
    "bell3a": (874957, 878430),
    "bell5": (8.96641e06, 8.96641e06),
    "blend2": (7.59898, 7.59898),
    "cap6000": (-2.45147e06, -2.45124e06),
    "dano3mip": (578.217, 696.051),
    "danoint": (62.8767, 65.6667),
    "dcmulti": (188172, 188182),
    "dsbmip": (-305.198, -305.198),
    "egout": (568.101, 568.101),
    "enigma": (0, 0),
    "fast0507": (173, 175),
    "fiber": (405935, 405935),
    "fixnet6": (3983, 3983),
    "gen": (112313, 112320),
    "gesa2_o": (2.57799e07, 2.57799e07),
    "gt2": (21166, 21166),
    "harp2": (-7.40777e07, -7.37744e07),
    "khb05250": (1.0694e08, 1.0694e08),
    "l152lav": (4722, 4722),
    "lseu": (1120, 1120),
    "markshare1": (-0, 48),
    "markshare2": (-0, 124),
    "mas74": (10634.2, 12343.4),
    "mas76": (39083.5, 40005.1),
    "misc03": (3360, 3360),
    "misc06": (12850.9, 12850.9),
    "misc07": (2010, 2895),
    "mitre": (115155, 115155),
    "mkc": (-564.696, -562.926),
    "mod008": (307, 307),
    "mod010": (6548, 6548),
    "mod011": (-5.45585e07, -5.45585e07),
    "modglob": (2.07405e07, 2.07405e07),
    "noswot": (-43, -41),
    "nw04": (16862, 16862),
    "p0033": (3089, 3089),
    "p0201": (7615, 7615),
    "p0282": (258411, 258411),
    "p0548": (8691, 8691),
    "p2756": (3124, 3124),
    "pk1": (0, 20),
    "pmp-iss-157": (986923.77543078, 986923.77543078),
    "pp08aCUTS": (7350, 7350),
    "pp08a": (7300.62, 7370),
    "qnet1": (16029.7, 16029.7),
    "qnet1_o": (16029.7, 16029.7),
    "rentacar": (3.03568e07, 3.03568e07),
    "rgn": (82.2, 82.2),
    "rout": (1023.55, 1079.19),
    "set1ch": (54537.8, 54537.8),
    "seymour": (415, 424),
    "stein27": (18, 18),
    "stein45": (24, 31),
    "swath": (381.166, 574.14),
    "vpm1": (20, 20),
    "vpm2": (13.75, 13.75),
}

TOL = 1e-4
MAX_NODES = 10

SOLVERS = [CBC]
if mip.gurobi.has_gurobi and "GUROBI_HOME" in environ:
    SOLVERS += [GUROBI]
if mip.highs.has_highs:
    SOLVERS += [HIGHS]

# check availability of test data
DATA_DIR = join(join(dirname(mip.__file__)[0:-3], "test"), "data")


print("DATA")
print(DATA_DIR)

# instance extensions
EXTS = [".mps", ".mps.gz", ".lp", ".lp.gz"]

# download here

INSTS = []
for exti in EXTS:
    INSTS = INSTS + glob(join(DATA_DIR, "*" + exti))


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver, instance", product(SOLVERS, INSTS))
def test_mip_file(solver: str, instance: str):
    """Tests optimization of MIP models stored in .mps or .lp files"""
    m = Model(solver_name=solver)

    # optional file for optimal LP basis
    bas_file = ""

    iname = ""
    for ext in EXTS:
        if instance.endswith(ext):
            bas_file = instance.replace(ext, ".bas")
            if not exists(bas_file):
                bas_file = ""
            iname = basename(instance.replace(ext, ""))
            break
    assert iname in BOUNDS.keys()

    lb = BOUNDS[iname][0]
    ub = BOUNDS[iname][1]
    assert lb <= ub + TOL
    has_opt = abs(ub - lb) <= TOL

    max_dif = max(max(abs(ub), abs(lb)) * 0.01, TOL)

    if solver == HIGHS and instance.endswith(".gz"):
        pytest.skip("HiGHS does not support .gz files.")

    m.read(instance)
    if bas_file:
       m.verbose = True
       m.read(bas_file)
       m.optimize(relax=True)
       print("Basis loaded!!! Obj value: %f" % m.objective_value)
    m.optimize(max_nodes=MAX_NODES)
    if m.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
        assert m.num_solutions >= 1
        m.check_optimization_results()
        assert m.objective_value >= lb - max_dif
        if has_opt and m.status == OptimizationStatus.OPTIMAL:
            assert abs(m.objective_value - ub) <= max_dif

    elif m.status == OptimizationStatus.NO_SOLUTION_FOUND:
        assert m.objective_bound <= ub + max_dif
    else:
        assert m.status not in [
            OptimizationStatus.INFEASIBLE,
            OptimizationStatus.INT_INFEASIBLE,
            OptimizationStatus.UNBOUNDED,
            OptimizationStatus.ERROR,
            OptimizationStatus.CUTOFF,
        ]
    assert m.objective_bound <= ub + max_dif
