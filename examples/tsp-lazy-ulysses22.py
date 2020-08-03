"""Example of a Branch-and-Cut implementation for the Traveling Salesman
Problem. Initially an incomplete formulation including only the degree
constraints is created. Sub-tour elimination constraints are generated
on-demand using a CutsGenerator that performs a depth-first search to identify
disconnected sub-routes."""

from typing import Tuple, Set, List
from math import floor, cos, acos
from itertools import product
from collections import defaultdict
from sys import stdout as out
from mip import Model, xsum, minimize, BINARY, ConstrsGenerator, CutPool


def subtour(N: Set, outa: defaultdict, node) -> List:
    """checks if a given node belongs to a disconnected sub-route and
    returns involved nodes"""
    queue = [node]
    visited = set(queue)
    while queue:
        n = queue.pop()
        for nl in outa[n]:
            if nl not in visited:
                queue.append(nl)
                visited.add(nl)

    if len(visited) != len(N):
        return list(visited)
    else:
        return []


class SubTourLazyGenerator(ConstrsGenerator):
    """generated sub-tour elimination constraints"""

    def __init__(self, xv):
        self._x = xv

    def generate_constrs(self, model: Model, depth: int = 0, npass: int = 0):
        x_, N, cp = model.translate(self._x), range(len(self._x)), CutPool()
        outa = [[j for j in N if x_[i][j].x >= 0.99] for i in N]

        for node in N:
            S = set(subtour(N, outa, node))
            if S:
                AS = [(i, j) for (i, j) in product(S, S) if i != j]
                cut = xsum(x_[i][j] for (i, j) in AS) <= len(S) - 1
                cp.add(cut)
        for cut in cp.cuts:
            model += cut


# constants as stated in TSPlib doc
# https://www.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/tsp95.pdf
PI = 3.141592
RRR = 6378.388


def rad(val: float) -> float:
    """converts to radians"""
    mult = 1.0
    if val < 0.0:
        mult = -1.0
        val = abs(val)

    deg = float(floor(val))
    minute = val - deg
    return (PI * (deg + 5 * minute / 3) / 180) * mult


def dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """computes geographical distance"""
    q1 = cos(p1[1] - p2[1])
    q2 = cos(p1[0] - p2[0])
    q3 = cos(p1[0] + p2[0])
    return int(
        floor(RRR * acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
    )


# coordinates of ulysses22 tsplib instance
coord = [
    (38.24, 20.42),
    (39.57, 26.15),
    (40.56, 25.32),
    (36.26, 23.12),
    (33.48, 10.54),
    (37.56, 12.19),
    (38.42, 13.11),
    (37.52, 20.44),
    (41.23, 09.10),
    (41.17, 13.05),
    (36.08, -5.21),
    (38.47, 15.13),
    (38.15, 15.35),
    (37.51, 15.17),
    (35.49, 14.32),
    (39.36, 19.56),
    (38.09, 24.36),
    (36.09, 23.00),
    (40.44, 13.57),
    (40.33, 14.15),
    (40.37, 14.23),
    (37.57, 22.56),
]

# latitude and longitude
coord = [(rad(x), rad(y)) for (x, y) in coord]

# distances in an upper triangular matrix

# number of nodes and list of vertices
n, V = len(coord), set(range(len(coord)))

# distances matrix
c = [[0 if i == j else dist(coord[i], coord[j]) for j in V] for i in V]

model = Model()

# binary variables indicating if arc (i,j) is used on the route or not
x = [[model.add_var(var_type=BINARY) for j in V] for i in V]

# objective function: minimize the distance
model.objective = minimize(xsum(c[i][j] * x[i][j] for i in V for j in V))

# constraint : leave each city only once
for i in V:
    model += xsum(x[i][j] for j in V - {i}) == 1

# constraint : enter each city only once
for i in V:
    model += xsum(x[j][i] for j in V - {i}) == 1

model.lazy_constrs_generator = SubTourLazyGenerator(x)

# optimizing
model.optimize(max_seconds=70)

# checking if a solution was found
if model.num_solutions:
    out.write(
        "route with total distance %g found: %s" % (model.objective_value, 0)
    )
    nc = 0
    while True:
        nc = [i for i in V if x[nc][i].x >= 0.99][0]
        out.write(" -> %s" % nc)
        if nc == 0:
            break
    out.write("\n")

# sanity tests
from mip import OptimizationStatus

if model.status == OptimizationStatus.OPTIMAL:
    assert round(model.objective_value) == 7013
elif model.status == OptimizationStatus.FEASIBLE:
    assert round(model.objective_value) >= 7013
else:
    assert model.objective_bound <= 7013 + 1e-7
model.check_optimization_results()
