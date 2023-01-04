import pytest
import mip
import mip.constants
from mip.conflict import ConflictFinder, IISFinderAlgorithm, ConflictRelaxer
import random
import numpy as np


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
    iis = cf.find_iis(method=IISFinderAlgorithm.ADDITIVE_ALGORITHM)
    iis_names = set([crt.name for crt in iis])
    assert set(["lower_bound", "upper_bound"]) == iis_names


def test_conflict_finder_iis_additive_method_two_options():
    mdl = mip.Model(name="infeasible_model_continuous")
    var_x = mdl.add_var(name="x", var_type=mip.CONTINUOUS, lb=-mip.INF, ub=mip.INF)
    mdl.add_constr(var_x >= 1, "lower_bound")
    mdl.add_constr(var_x <= 0, "upper_bound")
    mdl.add_constr(var_x <= -3, "upper_bound_2")

    cf = ConflictFinder(model=mdl)
    iis = cf.find_iis(method=IISFinderAlgorithm.ADDITIVE_ALGORITHM)
    iis_names = set([crt.name for crt in iis])
    assert (
        set(["lower_bound", "upper_bound"]) == iis_names
        or set(["lower_bound", "upper_bound_2"]) == iis_names
    )


def test_conflict_finder_feasible():
    mdl = mip.Model(name="feasible_model")
    var = mdl.add_var(name="x", var_type=mip.CONTINUOUS, lb=-mip.INF, ub=mip.INF)
    mdl.add_constr(var >= 1, "lower_bound")
    with pytest.raises(AssertionError, match="model is not linear infeasible"):
        cf = ConflictFinder(model=mdl)


def build_infeasible_cont_model(
    num_constraints: int = 10, num_infeasible_sets: int = 20
) -> mip.Model:
    # build an infeasible model, based on many redundant constraints
    mdl = mip.Model(name="infeasible_model_continuous")
    var = mdl.add_var(name="x", var_type=mip.CONTINUOUS, lb=-1000, ub=1000)

    for idx, rand_constraint in enumerate(np.linspace(1, 1000, num_constraints)):
        crt = mdl.add_constr(
            var >= rand_constraint,
            name="lower_bound_{0}".format(idx),
        )
        crt.priority = random.choice(list(mip.constants.ConstraintPriority)[1:])
        print(crt.priority)
        # logger.debug("added {} to the model".format(crt))

    num_constraint_inf = int(num_infeasible_sets / num_constraints)
    for idx, rand_constraint in enumerate(np.linspace(-1000, -1, num_constraint_inf)):
        crt = mdl.add_constr(var <= rand_constraint, name="upper_bound_{0}".format(idx))
        crt.priority = random.choice(list(mip.constants.ConstraintPriority)[:1])
        # logger.debug("added {} to the model".format(crt))

    mdl.emphasis = 1  # feasibility
    mdl.preprocess = 1  # -1  automatic, 0  off, 1  on.
    # mdl.pump_passes TODO configure to feasibility emphasis
    return mdl


def test_coflict_relaxer():
    # logger config
    # handler = logging.StreamHandler(sys.stdout)
    # logger.setLevel(logging.DEBUG)
    # logger.addHandler(handler)

    # create an infeasible model
    model = build_infeasible_cont_model()
    # logger.debug(model.status)
    model.optimize()
    # logger.debug(model.status)

    # find one IIS
    cf = ConflictFinder(model)
    iis = cf.find_iis()
    # logger.debug([crt.__str__() for crt in iis])

    # resolve a conflict
    cr = ConflictRelaxer(model)
    relaxed_model = cr.hierarchy_relaxer(relaxer_objective="min_abs_slack_val")
    # print(cr.slack_by_crt)
    # logger.debug(cr.slack_by_crt)
