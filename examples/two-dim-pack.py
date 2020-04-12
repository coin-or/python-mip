"""Example of modeling and solving the two dimensional level
packing problem in Python-MIP.
"""
from mip import Model, BINARY, minimize, xsum

#    0  1  2  3  4  5  6  7
w = [4, 3, 5, 2, 1, 4, 7, 3]  # widths
h = [2, 4, 1, 5, 6, 3, 5, 4]  # heights
n = len(w)
I = set(range(n))
S = [[j for j in I if h[j] <= h[i]] for i in I]
G = [[j for j in I if h[j] >= h[i]] for i in I]

# raw material width
W = 10

m = Model()

x = [{j: m.add_var(var_type=BINARY) for j in S[i]} for i in I]

m.objective = minimize(xsum(h[i] * x[i][i] for i in I))

# each item should appear as larger item of the level
# or as an item which belongs to the level of another item
for i in I:
    m += xsum(x[j][i] for j in G[i]) == 1

# represented items should respect remaining width
for i in I:
    m += xsum(w[j] * x[i][j] for j in S[i] if j != i) <= (W - w[i]) * x[i][i]

m.optimize()

for i in [j for j in I if x[j][j].x >= 0.99]:
    print(
        "Items grouped with {} : {}".format(
            i, [j for j in S[i] if i != j and x[i][j].x >= 0.99]
        )
    )

# sanity tests
from mip import OptimizationStatus

assert m.status == OptimizationStatus.OPTIMAL
assert round(m.objective_value) == 12
