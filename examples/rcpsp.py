"""Resource Constrained Project Scheduling Problem solver"""

from itertools import product
from mip import Model, xsum, BINARY

n = 10  # note there will be exactly 12 jobs (n=10 jobs plus the two 'dummy' ones)

p = [0, 3, 2, 5, 4, 2, 3, 4, 2, 4, 6, 0]

u = [[0, 0], [5, 1], [0, 4], [1, 4], [1, 3], [3, 2], [3, 1], [2, 4],
     [4, 0], [5, 2], [2, 5], [0, 0]]

c = [6, 8]

S = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 9], [2, 10], [3, 8], [4, 6],
     [4, 7], [5, 9], [5, 10], [6, 8], [6, 9], [7, 8], [8, 11], [9, 11], [10, 11]]

(R, J, T) = (range(len(c)), range(len(p)), range(sum(p)))

model = Model()

x = [[model.add_var(name="x({},{})".format(j, t), var_type=BINARY) for t in T] for j in J]

model.objective = xsum(t * x[n + 1][t] for t in T)

for j in J:
    model += xsum(x[j][t] for t in T) == 1

for (r, t) in product(R, T):
    model += (
        xsum(u[j][r] * x[j][t2] for j in J for t2 in range(max(0, t - p[j] + 1), t + 1))
        <= c[r])

for (j, s) in S:
    model += xsum(t * x[s][t] - t * x[j][t] for t in T) >= p[j]

model.optimize()

print("Schedule: ")
for (j, t) in product(J, T):
    if x[j][t].x >= 0.99:
        print("Job {}: begins at t={} and finishes at t={}".format(j, t, t+p[j]))
print("Makespan = {}".format(model.objective_value))


# sanity tests
from mip import OptimizationStatus

assert model.status == OptimizationStatus.OPTIMAL
assert abs(model.objective_value - 21) <= 1e-4
model.check_optimization_results()
