from milppy.model import *
import pdb

model = Model()

x = [ model.add_var(obj=1, name="x"+str(i)) for i in range(100) ]
c = model.add_constr(sum(x) == 5, "soma")

model.write('test.lp')
pdb.set_trace()
model.optimize()
