import pytest
import mip
from mip.conflict import ConflictFinder, IISFinderAlgorithm 
from examples import conflict_examples

def test_conflict_finder():
    mdl = mip.Model(name="infeasible_model_continuous")
    var = mdl.add_var(name="x", var_type=mip.CONTINUOUS, lb=-mip.INF, ub=mip.INF)
    mdl.add_constr(var >= 1, "lower_bound")
    mdl.add_constr(var <= 0, "upper_bound")

    cf = ConflictFinder(model=mdl)
    iis = cf.find_iis()
    iis_names = set([crt.name for crt in iis])
    assert set(["lower_bound", "upper_bound"]) == iis_names


def test_conflict_finder_iis():
    mdl = mip.Model(name="infeasible_model_continuous")
    var_x = mdl.add_var(name="x", var_type=mip.CONTINUOUS, lb=-mip.INF, ub=mip.INF)
    var_y = mdl.add_var(name="x", var_type=mip.CONTINUOUS, lb=-mip.INF, ub=mip.INF)
    mdl.add_constr(var_x >= 1, "lower_bound")
    mdl.add_constr(var_x <= 0, "upper_bound")
    mdl.add_constr(var_y <= -3, "upper_bound_2")

    cf = ConflictFinder(model=mdl)
    iis = cf.find_iis()
    iis_names = set([crt.name for crt in iis])
    assert set(["lower_bound", "upper_bound"]) == iis_names

def test_conflict_finder_iis_additive_method():
    mdl = mip.Model(name="infeasible_model_continuous")
    var_x = mdl.add_var(name="x", var_type=mip.CONTINUOUS, lb=-mip.INF, ub=mip.INF)
    var_y = mdl.add_var(name="x", var_type=mip.CONTINUOUS, lb=-mip.INF, ub=mip.INF)
    mdl.add_constr(var_x >= 1, "lower_bound")
    mdl.add_constr(var_x <= 0, "upper_bound")
    mdl.add_constr(var_y <= -3, "upper_bound_2")

    cf = ConflictFinder(model=mdl)
    iis = cf.find_iis(method = IISFinderAlgorithm.ADDITIVE_ALGORITHM)
    iis_names = set([crt.name for crt in iis])
    assert set(["lower_bound", "upper_bound"]) == iis_names

def test_conflict_finder_iis_additive_method_two_options():
    mdl = mip.Model(name="infeasible_model_continuous")
    var_x = mdl.add_var(name="x", var_type=mip.CONTINUOUS, lb=-mip.INF, ub=mip.INF)
    mdl.add_constr(var_x >= 1, "lower_bound")
    mdl.add_constr(var_x <= 0, "upper_bound")
    mdl.add_constr(var_x <= -3, "upper_bound_2")

    cf = ConflictFinder(model=mdl)
    iis = cf.find_iis(method = IISFinderAlgorithm.ADDITIVE_ALGORITHM)
    iis_names = set([crt.name for crt in iis])
    assert set(["lower_bound", "upper_bound"]) == iis_names or set(["lower_bound", "upper_bound_2"]) == iis_names

def test_conflict_finder_feasible():
    mdl = mip.Model(name="feasible_model")
    var = mdl.add_var(name="x", var_type=mip.CONTINUOUS, lb=-mip.INF, ub=mip.INF)
    mdl.add_constr(var >= 1, "lower_bound")
    with pytest.raises(AssertionError, match="model is not linear infeasible"):
        cf = ConflictFinder(model=mdl)

def test_examples():
    conflict_examples.main()
