"""Example where a binary program is created and the conflict graph
is inspected.
"""
from sys import stdout as out

from mip import Model, BINARY

m = m = Model(solver_name="cbc")

x = [m.add_var(var_type=BINARY) for i in range(6)]

m += -3 * x[0] + 4 * x[1] - 5 * x[2] + 6 * x[3] + 7 * x[4] + 8 * x[5] <= 2
m += x[0] + x[1] + x[3] >= 1

print(x[0])

cg = m.conflict_graph

for i in range(6):
    for v in range(2):
        out.write("conflicts for x[%d] == %g : " % (i, v))
        ca = cg.conflicting_assignments(x[i] == v)
        for j in ca[0]:
            out.write("x[%d] == 1  " % j.idx)
        for j in ca[1]:
            out.write("x[%d] == 0  " % j.idx)

        out.write("\n")


# print(cg.conflicting_assignments(x[1]))
