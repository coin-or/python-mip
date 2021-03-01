
import logging
import numpy as np
import sys
import random
from mip.conflict import ConflictFinder, ConflictRelaxer
import mip

# logger = logging.getLogger(__name__)
logger = logging.getLogger("conflict")


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
        logger.debug("added {} to the model".format(crt))

    num_constraint_inf = int(num_infeasible_sets / num_constraints)
    for idx, rand_constraint in enumerate(np.linspace(-1000, -1, num_constraint_inf)):
        crt = mdl.add_constr(
            var <= rand_constraint,
            name="upper_bound_{0}".format(idx)
        )
        crt.priority = random.choice(list(mip.constants.ConstraintPriority)[:1])
        logger.debug("added {} to the model".format(crt))

    mdl.emphasis = 1  # feasibility
    mdl.preprocess = 1  # -1  automatic, 0  off, 1  on.
    # mdl.pump_passes TODO configure to feasibility emphasis
    return mdl


def main():
    # logger config
    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # create an infeasible model
    model = build_infeasible_cont_model()
    logger.debug(model.status)
    model.optimize()
    logger.debug(model.status)

    # find one IIS
    cf = ConflictFinder(model)
    iis = cf.find_iis()
    logger.debug([crt.__str__() for crt in iis])

    # resolve a conflict
    cr = ConflictRelaxer(model)
    relaxed_model = cr.hierarchy_relaxer(relaxer_objective="min_abs_slack_val")
    print(cr.slack_by_crt)


if __name__ == "__main__":
    main()    