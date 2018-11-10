from milppy.model import *
import pdb

model = Model()

x = [model.add_var(obj=1, name="x%d" % i) for i in range(100)]
c = model.add_constr(quicksum(x) / 1.5 >= 5, "soma")
model.add_constr(x[0] + x[3] == 10)

model.set_objective(0.5 + 30 * x[0] + 100 * x[3])

model.write('test.lp')
model.optimize()
