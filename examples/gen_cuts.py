"""Example of use of general cutting planes in a small MIP."""

import sys
from mip import Model, INTEGER, maximize, CutType, OptimizationStatus

larger_diff = -1
best_cut = None

for ct in CutType:
    print("Trying cut type: {}".format(ct.name))

    m = Model()
    if m.solver_name.upper() in ["GRB", "GUROBI"]:
        print("This feature is currently not supported in Gurobi.")
    else:
        m.verbose = 0

        x = m.add_var_tensor(shape=(2, 1), name="x", var_type=INTEGER)

        m.objective = maximize(2 * x[0] + x[1])

        m += 7 * x[0] + x[1] <= 28
        m += -x[0] + 3 * x[1] <= 7
        m += -8 * x[0] - 9 * x[1] <= -32

        m.optimize(relax=True)
        olr = m.objective_value

        cp = m.generate_cuts([ct])
        if cp and cp.cuts:
            print("{} cuts generated:".format(len(cp.cuts)))
            for c in cp.cuts:
                print("  " + str(c))

        if cp.cuts:
            m += cp
            m.optimize(relax=True)

            print("Dual bound now: {}".format(m.objective_value))
            assert m.status == OptimizationStatus.OPTIMAL
            diff = m.objective_value - olr
            if diff > larger_diff:
                larger_diff = diff
                best_cut = ct

print("Best cut: {}".format(best_cut))
