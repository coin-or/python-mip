import mip

model = mip.Model(solver_name=mip.HIGHS)
model.verbose = 0
solver = model.solver

x = model.add_var(name="x")
y = model.add_var(name="y", lb=5, ub=23, var_type=mip.INTEGER)
z = model.add_var(name="z", var_type=mip.BINARY)

model += x + y == 99
model += x <= 99 * z
model += x + y + z >= 1

model.objective = mip.minimize(2*x - 3*y + 23)

status = model.optimize()
print(f"status: {status}")
print(f"objective value: {model.objective_value}")

# methods
print()
print(f"objective bound: {model.objective_bound}, {solver.get_objective_bound()}")
print(f"obj expr: {model.objective}, {solver.get_objective()}")
model.write("test.lp")

# internals
print()
print(f"Solver: {solver}")
print(f"cols: {solver.num_cols()}, rows: {solver.num_rows()}, nz: {solver.num_nz()}, int: {solver.num_int()},")
print(f"Var names: {solver._var_name}")
print(f"Var cols: {solver._var_col}")
print(f"Cons names: {solver._cons_name}")
print(f"Cons cols: {solver._cons_col}")
print(f"Sols: {solver._x}, {solver._rc}, {solver._pi}")

# changes
solver.relax()

# try again
status = model.optimize()
print()
print(f"Sols: {solver._x}, {solver._rc}, {solver._pi}")
