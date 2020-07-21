"""Example with clique merge: a formulation of a knapsack problem
with conflicting items where clique merge can improve it"""

from mip import Model, xsum, maximize, BINARY

I = set(range(1, 7))

# weigths
w = {1: -4, 2: 4, 3: 5, 4: 6, 5: 7, 6: 10}

# profits
p = {1: -2, 2: 8, 3: 10, 4: 12, 5: 13, 6: 13}

# capacity
c = 6

# conflicting items
C = ((2, 3), (2, 4), (3, 4), (2, 5))

m = Model()

x = {i: m.add_var("x({})".format(i), var_type=BINARY) for i in I}

m.objective = maximize(xsum(p[i] * x[i] for i in I))

m += xsum(w[i] * x[i] for i in I) <= c

for (i, j) in C:
    m += x[i] + x[j] <= 1

m.verbose = 0
m.write("b.lp")
m.optimize(relax=True)

print(
    "constraints before clique merging: {}. lower bound:"
    "{}.".format(len(m.constrs), m.objective_value)
)

m.clique_merge()

m.optimize(relax=True)

print(
    "constraints after clique merging: {}. lower bound:"
    "{}.".format(len(m.constrs), m.objective_value)
)

m.optimize()

print("optimal: {}".format(m.objective_value))


# sanity tests
from mip import OptimizationStatus

assert m.status == OptimizationStatus.OPTIMAL
if m.solver_name.upper() == "CBC":
    assert m.num_rows == 2
assert abs(m.objective_value - 12) <= 1e-4
