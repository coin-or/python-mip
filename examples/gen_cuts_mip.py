"""This example reads a MIP (in .lp or .mps), solves its linear programming
relaxation and then tests the impact of adding different types of cutting
planes. In the end, the it informs which cut generator produced the best bound
improvement."""

from textwrap import shorten
import sys
from mip import Model, CutType, OptimizationStatus
import mip

lp_path = ""

#  using test data
lp_path = mip.__file__.replace("mip/__init__.py", "test/data/1443_0-9.lp").replace(
    "mip\\__init__.py", "test\\data\\1443_0-9.lp"
)

m = Model()
if m.solver_name.upper() in ["GRB", "GUROBI"]:
    print("This feature is currently not supported in Gurobi.")
else:
    m.read(lp_path)

    m.verbose = 0
    m.optimize(relax=True)
    print("Original LP bound: {}".format(m.objective_value))

    best_impr = -1
    best_cut = ""

    for ct in CutType:
        print()
        m2 = m.copy()
        m2.verbose = 0
        m2.optimize(relax=True)
        assert (
            m2.status == OptimizationStatus.OPTIMAL
            and abs(m2.objective_value - m.objective_value) <= 1e-4
        )
        print("Searching for violated {} inequalities ...".format(ct.name))
        cp = m2.generate_cuts([ct])
        if cp and cp.cuts:
            print("{} cuts found:".format(len(cp.cuts)))
            for c in cp.cuts[0 : min(10, len(cp.cuts))]:
                print("  {}".format(shorten(str(c), width=90, placeholder="...")))
            m2 += cp
            m2.optimize(relax=True)
            perc_impr = (
                abs(m2.objective_value - m.objective_value)
                / max(abs(m2.objective_value), abs(m.objective_value))
            ) * 100

            if perc_impr > best_impr:
                best_impr = perc_impr
                best_cut = ct

            print(
                "Linear programming relaxation bound now: %g, improvement of %.2f"
                % (m2.objective_value, perc_impr)
            )
        else:
            continue

    print("Best cut:  {}   improved dual bound: {:.2f} ".format(best_cut, best_impr))
