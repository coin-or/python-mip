"""Resource constrained project scheduling problem Model
creation benchmark
"""

from sys import argv
from typing import Tuple, List
from time import time
from os.path import basename
from mip import Model, minimize, xsum, CONTINUOUS, BINARY


def read_rcpsp(
    file_name: str,
) -> Tuple[
    List[int],
    List[int],
    List[Tuple[int, int]],
    List[int],
    List[List[int]],
    List[int],
]:
    """Reads a RCPSP file and returns
       J jobs
       d duration
       S list of successors for a node
       c resource capacity
       r resource usage
       EST for each job
    """

    f = open(file_name, "r")
    section = 0
    nl = 1
    S = []
    for l in f:
        nl += 1
        if section == 0:
            if "jobs (" in l:
                J = list(range(int(l.split(":")[1].strip())))
                EST = [0 for j in J]
                r = [[] for j in J]
                d = [0 for j in J]
                section += 1
        elif section == 1:
            if "horizon" in l:
                t = int(l.split(":")[1].strip())
                section += 1
        elif section == 2:
            if "jobnr. " in l:
                section += 1
        elif section == 3:
            s = l.replace("  ", " ")
            while "  " in s:
                s = s.replace("  ", " ")
            s = s.lstrip().rstrip()
            v = s.split(" ")
            j = int(v[0]) - 1
            for s in [(int(vl) - 1) for vl in v[3:]]:
                S.append((j, s))
            if j == len(J) - 1:
                section += 1
        elif section == 4:
            if "----" in l:
                section += 1
        elif section == 5:
            s = l.replace("  ", " ")
            while "  " in s:
                s = s.replace("  ", " ")
            s = s.lstrip().rstrip()
            v = s.split(" ")
            j, dur = int(v[0]) - 1, int(v[2])
            d[j] = dur
            r[j] = [int(vl) for vl in v[3:]]
            if j == len(J) - 1:
                section += 1
        elif section == 6:
            if "R 1" in l:
                section += 1
        elif section == 7:
            s = l.replace("  ", " ")
            while "  " in s:
                s = s.replace("  ", " ")
            s = s.lstrip().rstrip()
            v = s.split(" ")
            c = [int(vl) for vl in v]
            section += 1

    f.close()

    # computing EST
    for _ in J:
        for (u, v) in S:
            EST[v] = max(EST[u] + d[u], EST[v])

    return (J, d, S, c, r, EST)


def create_mip(solver, J, dur, S, c, r, EST, relax=False) -> Model:
    """Creates a mip model to solve the RCPSP"""
    NR = len(c)
    mip = Model(solver_name=solver)
    sd = sum(dur[j] for j in J)
    vt = CONTINUOUS if relax else BINARY
    x = [
        {
            t: mip.add_var("x(%d,%d)" % (j, t), var_type=vt)
            for t in range(EST[j], sd + 1)
        }
        for j in J
    ]
    TJ = [set(x[j].keys()) for j in J]
    T = set()
    for j in J:
        T = T.union(TJ[j])

    mip.objective = minimize(xsum(t * x[J[-1]][t] for t in TJ[-1]))

    # one time per job
    for j in J:
        mip += xsum(x[j][t] for t in TJ[j]) == 1, "selTime(%d)" % j

    # precedences
    for (u, v) in S:
        mip += (
            xsum(t * x[v][t] for t in TJ[v])
            >= xsum(t * x[u][t] for t in TJ[u]) + dur[u],
            "prec(%d,%d)" % (u, v),
        )

    # resource usage
    for t in T:
        for ir in range(NR):
            mip += (
                xsum(
                    r[j][ir] * x[j][tl]
                    for j in J[1:-1]
                    for tl in TJ[j].intersection(
                        set(range(t - dur[j] + 1, t + 1))
                    )
                )
                <= c[ir],
                "resUsage(%d,%d)" % (ir, t),
            )

    return mip


if len(argv) < 4:
    print("usage: rcpsp instance solver numberTimeMeasures")

J, d, S, c, r, EST = read_rcpsp(argv[1])
ntimes = int(argv[3].strip())
st = time()
for _ in range(ntimes):
    mip = create_mip(argv[2], J, d, S, c, r, EST, True)
ed = time()
secs = (ed - st) / ntimes
iname = basename(argv[1])
print(
    "{},{},{},{},{},{}".format(
        iname, argv[2], mip.num_cols, mip.num_rows, mip.num_nz, secs
    )
)
