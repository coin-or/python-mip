"""Example where a binary program is created and the conflict graph
is inspected.
"""
from itertools import product
from sys import stdout as out
from mip import Model, BINARY


m = m = Model(solver_name="cbc")

N = range(1, 7)

x = {i: m.add_var(var_type=BINARY, name="x(%d)" % i) for i in N}

m += -3 * x[1] + 4 * x[2] - 5 * x[3] + 6 * x[4] + 7 * x[5] + 8 * x[6] <= 2
m += x[1] + x[2] + x[4] >= 1

cg = m.conflict_graph

for i, v in product(N, range(2)):
    out.write("conflicts for x[%d] == %g : " % (i, v))
    ca = cg.conflicting_assignments(x[i] == v)
    for j in ca[0]:
        out.write("%s == 1  " % j.name)
    for j in ca[1]:
        out.write("%s == 0  " % j.name)

    out.write("\n")

# sanity checks
confs = {
    (i, v): cg.conflicting_assignments(x[i] == v)
    for (i, v) in product(N, range(2))
}
# conflicts with complement
for i in N:
    assert cg.conflicting(x[i] == 1, x[i] == 0)
    assert x[i] in confs[i, 1][1]
    assert x[i] in confs[i, 0][0]
# other conflicts to test
test_conf = [((2, 1), (5, 1)), ((2, 1), (6, 1)), ((2, 1), (2, 0))]
for c in test_conf:
    assert cg.conflicting(x[c[0][0]] == c[0][1], x[c[1][0]] == c[1][1])

test_no_conf = [((2, 1), (4, 1))]
for c in test_no_conf:
    assert not cg.conflicting(x[c[0][0]] == c[0][1], x[c[1][0]] == c[1][1])
