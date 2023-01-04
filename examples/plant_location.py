r"""Example with Special Ordered Sets (SOS) type 1 and 2.
Plant location: one industry plans to install two plants in two different
regions, one to the west and another to the east. It must decide also the
production capacity of each plant and allocate clients to plants
in order to minimize shipping costs. We'll use SOS of type 1 to ensure that
only one of the plants in each region has a non-zero production capacity.
The cost :math:`f(z)` of building a plant with capacity :math:`z` grows according
to the non-linear function :math:`f(z)=1520 \log z`. Type 2 SOS will be used to
model the cost of installing each one of the plants.
"""

import sys

# Workaround for issues with python not being installed as a framework on mac
# by using a different backend.
if sys.platform == "darwin":  # OS X
    import matplotlib as mpl
    mpl.use('Agg')
    del mpl

import matplotlib.pyplot as plt
from math import sqrt, log
from itertools import product
from mip import Model, xsum, minimize, OptimizationStatus

# possible plants
F = [1, 2, 3, 4, 5, 6]

# possible plant installation positions
pf = {1: (1, 38), 2: (31, 40), 3: (23, 59), 4: (76, 51), 5: (93, 51), 6: (63, 74)}

# maximum plant capacity
c = {1: 1955, 2: 1932, 3: 1987, 4: 1823, 5: 1718, 6: 1742}

# clients
C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# position of clients
pc = {1: (94, 10), 2: (57, 26), 3: (74, 44), 4: (27, 51), 5: (78, 30), 6: (23, 30), 
      7: (20, 72), 8: (3, 27), 9: (5, 39), 10: (51, 1)}

# demands
d = {1: 302, 2: 273, 3: 275, 4: 266, 5: 287, 6: 296, 7: 297, 8: 310, 9: 302, 10: 309}

# plotting possible plant locations
for i, p in pf.items():
    plt.scatter((p[0]), (p[1]), marker="^", color="purple", s=50)
    plt.text((p[0]), (p[1]), "$f_%d$" % i)

# plotting location of clients
for i, p in pc.items():
    plt.scatter((p[0]), (p[1]), marker="o", color="black", s=15)
    plt.text((p[0]), (p[1]), "$c_{%d}$" % i)

plt.text((20), (78), "Region 1")
plt.text((70), (78), "Region 2")
plt.plot((50, 50), (0, 80))

dist = {(f, c): round(sqrt((pf[f][0] - pc[c][0]) ** 2 + (pf[f][1] - pc[c][1]) ** 2), 1)
        for (f, c) in product(F, C) }

m = Model()

z = {i: m.add_var(ub=c[i]) for i in F}  # plant capacity

# Type 1 SOS: only one plant per region
for r in [0, 1]:
    # set of plants in region r
    Fr = [i for i in F if r * 50 <= pf[i][0] <= 50 + r * 50]
    m.add_sos([(z[i], i - 1) for i in Fr], 1)

# amount that plant i will supply to client j
x = {(i, j): m.add_var() for (i, j) in product(F, C)}

# satisfy demand
for j in C:
    m += xsum(x[(i, j)] for i in F) == d[j]

# SOS type 2 to model installation costs for each installed plant
y = {i: m.add_var() for i in F}
for f in F:
    D = 6  # nr. of discretization points, increase for more precision
    v = [c[f] * (v / (D - 1)) for v in range(D)]  # points
    # non-linear function values for points in v
    vn = [0 if k == 0 else 1520 * log(v[k]) for k in range(D)]  
    # w variables
    w = [m.add_var() for v in range(D)]
    m += xsum(w) == 1  # convexification
    # link to z vars
    m += z[f] == xsum(v[k] * w[k] for k in range(D))
    # link to y vars associated with non-linear cost
    m += y[f] == xsum(vn[k] * w[k] for k in range(D))
    m.add_sos([(w[k], v[k]) for k in range(D)], 2)

# plant capacity
for i in F:
    m += z[i] >= xsum(x[(i, j)] for j in C)

# objective function
m.objective = minimize(
    xsum(dist[i, j] * x[i, j] for (i, j) in product(F, C)) + xsum(y[i] for i in F) )

m.optimize()

plt.savefig("location.pdf")

if m.num_solutions:
    print("Solution with cost {} found.".format(m.objective_value))
    print("Facilities capacities: {} ".format([z[f].x for f in F]))
    print("Facilities cost: {}".format([y[f].x for f in F]))

    # plotting allocations
    for (i, j) in [(i, j) for (i, j) in product(F, C) if x[(i, j)].x >= 1e-6]:
        plt.plot(
            (pf[i][0], pc[j][0]), (pf[i][1], pc[j][1]), linestyle="--", color="darkgray"
        )

    plt.savefig("location-sol.pdf")

# sanity checks
opt = 99733.94905406
if m.status == OptimizationStatus.OPTIMAL:
    assert abs(m.objective_value - opt) <= 0.01
elif m.status == OptimizationStatus.FEASIBLE:
    assert m.objective_value >= opt - 0.01
else:
    assert m.status not in [OptimizationStatus.INFEASIBLE, OptimizationStatus.UNBOUNDED]
