from milppy.model import *
import pdb

model = Model()

x = [model.add_var(obj=1, name="x%d" % i) for i in range(100)]
c = model.add_constr(quicksum(x) / 1.5 >= 5, "soma")

model.set_objective(x[0])

model.write('test.lp')
model.optimize()

# from gurobipy import *

# model = Model()
