"""This example reads a MIP (in .lp or .mps), solves its linear programming
relaxation and then tests the impact of adding different types of cutting
planes."""

import sys
from mip import Model, features, compute_features
import mip

lp_path = ""

#  using test data, replace with your instance
lp_path = mip.__file__.replace("mip/__init__.py", "test/data/1443_0-9.lp").replace(
    "mip\\__init__.py", "test\\data\\1443_0-9.lp"
)

m = Model()
if m.solver_name.upper() in ["GRB", "GUROBI"]:
    print("This feature is only supported in CBC.")
else:
    m.read(lp_path)

    print("instance features:")
    X = compute_features(m)
    for i, fn in enumerate(features()):
        print("%s: %g" % (fn, X[i]))
