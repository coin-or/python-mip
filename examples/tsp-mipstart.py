"""Example that solves the Traveling Salesman Problem using the simple compact
formulation presented in Miller, C.E., Tucker, A.W and Zemlin, R.A. "Integer
Programming Formulation of Traveling Salesman Problems". Journal of the ACM
7(4). 1960."""

from itertools import product
from sys import stdout as out
from mip import Model, xsum, minimize, BINARY
import random as rnd
from typing import List

# names of places to visit
places = ['Antwerp', 'Bruges', 'C-Mine', 'Dinant', 'Ghent',
          'Grand-Place de Bruxelles', 'Hasselt', 'Leuven',
          'Mechelen', 'Mons', 'Montagne de Bueren', 'Namur',
          'Remouchamps', 'Waterloo']

# distances in an upper triangular matrix
dists = [[83, 81, 113, 52, 42, 73, 44, 23, 91, 105, 90, 124, 57],
         [161, 160, 39, 89, 151, 110, 90, 99, 177, 143, 193, 100],
         [90, 125, 82, 13, 57, 71, 123, 38, 72, 59, 82],
         [123, 77, 81, 71, 91, 72, 64, 24, 62, 63],
         [51, 114, 72, 54, 69, 139, 105, 155, 62],
         [70, 25, 22, 52, 90, 56, 105, 16],
         [45, 61, 111, 36, 61, 57, 70],
         [23, 71, 67, 48, 85, 29],
         [74, 89, 69, 107, 36],
         [117, 65, 125, 43],
         [54, 22, 84],
         [60, 44],
         [97],
         []]

# number of nodes and list of vertices
n, V = len(dists), range(len(dists))

# distances matrix
c = [[0 if i == j
      else dists[i][j-i-1] if j > i
      else dists[j][i-j-1]
      for j in V] for i in V]

model = Model()

# binary variables indicating if arc (i,j) is used on the route or not
x = [[model.add_var(var_type=BINARY) for j in V] for i in V]

# continuous variable to prevent subtours: each city will have a
# different sequential id in the planned route except the first one
y = [model.add_var() for i in V]

# objective function: minimize the distance
model.objective = minimize(xsum(c[i][j]*x[i][j] for i in V for j in V))

# constraint : leave each city only once
for i in V:
    model += xsum(x[i][j] for j in set(V) - {i}) == 1

# constraint : enter each city only once
for i in V:
    model += xsum(x[j][i] for j in set(V) - {i}) == 1

# subtour elimination
for (i, j) in set(product(set(V) - {0}, set(V) - {0})):
    model += y[i] - (n+1)*x[i][j] >= y[j]-n

# running a best insertion heuristic to obtain an initial feasible solution:
# test every node j not yet inserted in the route at every intermediate
# position p and select the pair (j, p) that results in the smallest cost
# increase
seq = [0, max((c[0][j], j) for j in V)[1]] + [0]
Vout = set(V)-set(seq)
while Vout:
    (j, p) = min([(c[seq[p]][j] + c[j][seq[p+1]], (j, p)) for j, p in
                  product(Vout, range(len(seq)-1))])[1]

    seq = seq[:p+1]+[j]+seq[p+1:]
    assert(seq[-1] == 0)
    Vout = Vout - {j}
cost = sum(c[seq[i]][seq[i+1]] for i in range(len(seq)-1))
print('route with cost %g built' % cost)


# function to evaluate the cost of swapping two positions in a route in
# constant time
def delta(d: List[List[float]], S: List[int], p1: int, p2: int) -> float:
    p1, p2 = min(p1, p2), max(p1, p2)
    e1, e2 = S[p1], S[p2]
    if p1 == p2:
        return 0
    elif abs(p1-p2) == 1:
        return ((d[S[p1-1]][e2] + d[e2][e1] + d[e1][S[p2+1]])
                - (d[S[p1-1]][e1] + d[e1][e2] + d[e2][S[p2+1]]))
    else:
        return (
        (d[S[p1-1]][e2] + d[e2][S[p1+1]] + d[S[p2-1]][e1] + d[e1][S[p2+1]])
        - (d[S[p1-1]][e1] + d[e1][S[p1+1]] + d[S[p2-1]][e2] + d[e2][S[p2+1]]))


# applying the Late Acceptance Hill Climbing
rnd.seed(0)
L = [cost for i in range(50)]
sl, cur_cost, best = seq.copy(), cost, cost
for it in range(int(1e7)):
    (i, j) = rnd.randint(1, len(sl)-2), rnd.randint(1, len(sl)-2)
    dlt = delta(c, sl, i, j)
    if cur_cost + dlt <= L[it % len(L)]:
        sl[i], sl[j], cur_cost = sl[j], sl[i], cur_cost + dlt
        if cur_cost < best:
            seq, best = sl.copy(), cur_cost
    L[it % len(L)] = cur_cost

print('improved cost %g' % best)

model.start = [(x[seq[i]][seq[i+1]], 1) for i in range(len(seq)-1)]
# optimizing
model.optimize(max_seconds=30)

# checking if a solution was found
if model.num_solutions:
    out.write('route with total distance %g found: %s'
              % (model.objective_value, places[0]))
    nc = 0
    while True:
        nc = [i for i in V if x[nc][i].x >= 0.99][0]
        out.write(' -> %s' % places[nc])
        if nc == 0:
            break
    out.write('\n')
model.check_optimization_results()
