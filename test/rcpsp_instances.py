"""This code generates a set of non-trivial Resource Constrained Project
Scheduling (RCPSP) instances. Non-trivial in the sense that for a compact
formulation the bound of the linear programming relaxation is far from the
optimal solution cost. Thus, cuts and branching are necessary to solve the
MIP"""
from random import randint
import json
from math import ceil
from mip import OptimizationStatus
from mip_rcpsp import create_mip

Dmin = 1
Dmax = 10
MIN_SUCC = 1
MAX_SUCC = 3
NR = 3  # number of resources
MAXRU = 15


def gen_inst(nj):
    """Generates RCPSP instances"""
    J = [0] + [i for i in range(1, nj + 1)] + [nj + 1]
    dur = [0] + [randint(Dmin, Dmax + 1) for j in range(nj)] + [0]
    S = []
    SJ = [[]] * len(J)
    PJ = [[]] * len(J)
    S += [(0, v) for v in J[1:-1]]
    S += [(u, J[-1]) for u in J[1:-1]]
    for j in range(1, nj + 1):
        for i in range(randint(1, 3)):  # successors
            u = randint(1, nj - 1)
            v = randint(u + 1, nj)
            u += 1
            v += 1
            SJ[u] += [v]
            PJ[v] += [u]
            S = S + [(u, v)]

    S = list(set(S))

    r = [
        ([0] + [randint(1, MAXRU) for j in range(nj)] + [0])
        for ir in range(NR)
    ]

    minr = [max(r[ir][j] for j in J) for ir in range(NR)]
    maxr = [sum(r[ir][j] for j in J) for ir in range(NR)]

    if nj <= 10:
        perc = 0.2
    elif nj <= 18:
        perc = 0.1
    else:
        perc = 0.05

    c = [ceil(minr[ir] + (maxr[ir] - minr[ir]) * perc) for ir in range(NR)]

    EST = [0 for j in J]
    for j in J:
        for (u, v) in S:
            EST[v] = max(EST[u] + dur[u], EST[v])

    inst = {"J": J, "d": dur, "S": S, "c": c, "r": r, "EST": EST}

    return inst


for nj in range(5, 25):
    print("Generating instance for %d jobs" % nj)
    if nj < 10:
        tries = 300
        ni = 10
    elif nj < 15:
        tries = 150
        ni = 5
    else:
        ni = 2
        tries = 50
    for ii in range(ni):
        best_inst = None
        best_score = -float("inf")
        best_z_relax = None
        best_z_ub = None
        best_z_lb = None

        for tr in range(tries):
            inst = gen_inst(nj)
            J = inst["J"]
            dur = inst["d"]
            c = inst["c"]
            r = inst["r"]
            S = inst["S"]
            EST = inst["EST"]
            mip = create_mip("gurobi", J, dur, S, c, r, EST)
            mip.verbose = 0
            mip.relax()
            mip.optimize()
            assert mip.status == OptimizationStatus.OPTIMAL
            z_relax = mip.objective_value
            mip = create_mip("gurobi", J, dur, S, c, r, EST)
            mip.verbose = 0
            mip.max_nodes = 512
            mip.optimize()
            z_lb = mip.objective_bound
            z_ub = mip.objective_value
            assert mip.status in [
                OptimizationStatus.OPTIMAL,
                OptimizationStatus.FEASIBLE,
            ]
            sum_dur = sum(dur[i] for i in J)
            score = (
                (z_ub - z_relax) / sum_dur
                + 2 * (z_ub - z_relax)
                + 5 * (z_ub - z_lb)
            )
            assert score >= -1e-10
            if score > best_score:
                best_score = score
                best_inst = inst
                best_z_relax = z_relax
                best_z_lb = z_lb
                best_z_ub = z_ub
        inst = best_inst
        with open("rcpsp-%d-%d.json" % (nj, ii + 1), "w") as outfile:
            json.dump(inst, outfile, indent=4)
