import mip

model = mip.Model(solver_name=mip.HIGHS)

x = model.add_var(name="x")
y = model.add_var(name="y", lb=5, ub=23, var_type=mip.INTEGER)
z = model.add_var(name="z", var_type=mip.BINARY)
