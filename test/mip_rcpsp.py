"""Creates a compact MIP formulation for the resource constrained project
scheduling problem"""
from mip import Model, BINARY, CONTINUOUS, xsum, minimize, MINIMIZE, maximize


def create_mip(solver, J, dur, S, c, r, EST, relax=False, sense=MINIMIZE):
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

    if sense == MINIMIZE:
        mip.objective = minimize(xsum(t * x[J[-1]][t] for t in TJ[-1]))
    else:
        mip.objective = maximize(xsum(t * x[J[-1]][t] for t in TJ[-1]))

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
                    r[ir][j] * x[j][tl]
                    for j in J[1:-1]
                    for tl in TJ[j].intersection(
                        set(range(t - dur[j] + 1, t + 1))
                    )
                )
                <= c[ir],
                "resUsage(%d,%d)" % (ir, t),
            )

    return mip
