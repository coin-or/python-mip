"""Tests for MILP Model for Failures and Attacks on a BFT Blockchain Protocol
more details at https://github.com/NeoResearch/milp_bft_failures_attacks"""

from os import environ
from itertools import product
import pytest
from mip import (
    Model,
    BINARY,
    INTEGER,
    xsum,
    OptimizationStatus,
    maximize,
    CBC,
    GUROBI,
)

TOL = 1e-4

SOLVERS = [CBC]
if "GUROBI_HOME" in environ:
    SOLVERS += [GUROBI]

# for each pair o N, tMax, the expected: (optimal, columns, rows, non-zeros)
PDATA = {
    (5, 7): (1005.0, 4633, 6681, 35851),
    (7, 6): (1007.0, 10202, 15076, 76116),
    (5, 20): (1005.0, 12758, 17666, 221322),
    (6, 8): (6006.0, 8721, 12122, 72839),
    (6, 10): (6006.0, 10809, 14858, 106685),
    (8, 9): (1008.0, 21963, 30870, 214172),
    (7, 15): (1007.0, 24755, 34795, 363846),
    (8, 13): (1008.0, 31435, 43414, 406528),
    (10, 20): (1010.0, 91013, 124178, 1733927),
    (15, 15): (1015.0, 221643, 297359, 3330422),
}


def create_model(solver, N, tMax):
    """creates the dbft mip"""
    f = int((N - 1) / 3)
    M = 2 * f + 1
    R = set(range(1, N + 1))
    R_OK = set(range(1, M + 1))
    V = set(range(1, N + 1))
    T = set(range(1, tMax + 1))
    m = Model(solver_name=solver)
    Primary = {
        (i, v): m.add_var("Primary(%s,%s)" % (i, v), var_type=BINARY)
        for (i, v) in product(R, V)
    }
    SendPrepReq = {
        (t, i, v): m.add_var("SendPrepReq(%s,%s,%s)" % (t, i, v), var_type=BINARY)
        for (t, i, v) in product(T, R, V)
    }
    SendPrepRes = {
        (t, i, v): m.add_var("SendPrepRes(%s,%s,%s)" % (t, i, v), var_type=BINARY)
        for (t, i, v) in product(T, R, V)
    }
    SendCommit = {
        (t, i, v): m.add_var("SendCommit(%s,%s,%s)" % (t, i, v), var_type=BINARY)
        for (t, i, v) in product(T, R, V)
    }
    SendCV = {
        (t, i, v): m.add_var("SendCV(%s,%s,%s)" % (t, i, v), var_type=BINARY)
        for (t, i, v) in product(T, R, V)
    }
    BlockRelay = {
        (t, i, v): m.add_var("BlockRelay(%s,%s,%s)" % (t, i, v), var_type=BINARY)
        for (t, i, v) in product(T, R, V)
    }
    RecvPrepReq = {
        (t, i, j, v): m.add_var(
            "RecvPrepReq(%s,%s,%s,%s)" % (t, i, j, v), var_type=BINARY
        )
        for (t, i, j, v) in product(T, R, R, V)
    }
    RecvPrepResp = {
        (t, i, j, v): m.add_var(
            "RecvPrepResp(%s,%s,%s,%s)" % (t, i, j, v), var_type=BINARY,
        )
        for (t, i, j, v) in product(T, R, R, V)
    }
    RecvCommit = {
        (t, i, j, v): m.add_var(
            "RecvCommit(%s,%s,%s,%s)" % (t, i, j, v), var_type=BINARY
        )
        for (t, i, j, v) in product(T, R, R, V)
    }
    RecvCV = {
        (t, i, j, v): m.add_var("RecvCV(%s,%s,%s,%s)" % (t, i, j, v), var_type=BINARY)
        for (t, i, j, v) in product(T, R, R, V)
    }

    totalBlockRelayed = m.add_var("totalBlockRelayed", var_type=INTEGER)
    blockRelayed = {v: m.add_var("blockRelayed(%s)" % v, var_type=BINARY) for v in V}
    lastRelayedBlock = m.add_var("lastRelayedBlock", var_type=INTEGER)
    numberOfRounds = m.add_var("numberOfRounds", var_type=INTEGER)
    changeViewRecvPerNodeAndView = {
        (r, v): m.add_var(
            "changeViewRecvPerNodeAndView(%s,%s)" % (r, v), var_type=INTEGER
        )
        for (r, v) in product(R, V)
    }
    prepReqSendPerNodeAndView = {
        (r, v): m.add_var("prepReqSendPerNodeAndView(%s,%s)" % (r, v), var_type=INTEGER)
        for (r, v) in product(R, V)
    }
    prepRespSendPerNodeAndView = {
        (r, v): m.add_var("prepRespSendPerNodeAndView(%s,%s)" % (r, v), var_type=INTEGER)
        for (r, v) in product(R, V)
    }
    commitSendPerNodeAndView = {
        (r, v): m.add_var("commitSendPerNodeAndView(%s,%s)" % (r, v), var_type=INTEGER)
        for (r, v) in product(R, V)
    }
    changeViewSendPerNodeAndView = {
        (r, v): m.add_var(
            "changeViewSendPerNodeAndView(%s,%s)" % (r, v), var_type=INTEGER
        )
        for (r, v) in product(R, V)
    }
    prepReqRecvPerNodeAndView = {
        (r, v): m.add_var("prepReqRecvPerNodeAndView(%s,%s)" % (r, v), var_type=INTEGER)
        for (r, v) in product(R, V)
    }
    prepRespRecvPerNodeAndView = {
        (r, v): m.add_var("prepRespRecvPerNodeAndView(%s,%s)" % (r, v), var_type=INTEGER)
        for (r, v) in product(R, V)
    }
    commitRecvPerNodeAndView = {
        (r, v): m.add_var("commitRecvPerNodeAndView(%s,%s)" % (r, v), var_type=INTEGER)
        for (r, v) in product(R, V)
    }
    blockRelayPerNodeAndView = {
        (r, v): m.add_var("blockRelayPerNodeAndView(%s,%s)" % (r, v), var_type=INTEGER)
        for (r, v) in product(R, V)
    }

    for (i, v) in product(R, V):
        m += SendPrepReq[1, i, v] == 0, "initSendPrepReq(%s,%s)" % (i, v)
        m += SendPrepRes[1, i, v] == 0, "initSendPrepRes(%s,%s)" % (i, v)
        m += SendCommit[1, i, v] == 0, "initSendCommit(%s,%s)" % (i, v)
        m += SendCV[1, i, v] == 0, "initSendCV(%s,%s)" % (i, v)
        m += BlockRelay[1, i, v] == 0, "initBlockRelay(%s,%s)" % (i, v)

    for (i, j, v) in product(R, R, V):
        m += RecvPrepReq[1, i, j, v] == 0, "initRecvPrepReq(%s,%s,%s)" % (i, j, v)
        m += RecvPrepResp[1, i, j, v] == 0, "initRecvPrepRes(%s,%s,%s)" % (i, j, v)
        m += RecvCommit[1, i, j, v] == 0, "initRecvCommit(%s,%s,%s)" % (i, j, v)
        m += RecvCV[1, i, j, v] == 0, "initRecvCV(%s,%s,%s)" % (i, j, v)

    m += xsum(Primary[i, 1] for i in R) == 1, "consensusShouldStart"

    for v in V:
        m += xsum(Primary[i, v] for i in R) <= 1, "singlePrimaryEveryView(%s)" % v

    for i in R:
        m += xsum(Primary[i, v] for v in V) <= 1, "primaryOO(%s)" % i

    for v in V - {1}:
        m += (
            xsum(Primary[i, v] * (v - 1) for i in R)
            <= xsum(Primary[i, v2] for i in R for v2 in V if v2 < v),
            "avoidJumpingViews(%s)" % v,
        )

    for (i, v) in product(R, V - {1}):
        m += (
            Primary[i, v] <= (1 / M) * changeViewRecvPerNodeAndView[i, v - 1],
            "nextPrimaryOnlyIfEnoughCV(%s,%s)" % (i, v),
        )

    for (t, i, v) in product(T - {1}, R, V):
        m += (
            SendPrepRes[t, i, v]
            <= xsum(RecvPrepReq[t2, i, j, v] for t2 in T if t2 <= t for j in R),
            "prepRespSendOptionally(%s,%s,%s)" % (t, i, v),
        )
        m += (
            SendCommit[t, i, v]
            <= (1 / M)
            * xsum(RecvPrepResp[t2, i, j, v] for t2 in T if t2 <= t for j in R),
            "commitSentIfMPrepRespOptionally(%s,%s,%s)" % (t, i, v),
        )
        m += (
            BlockRelay[t, i, v]
            <= (1 / M) * xsum(RecvCommit[t2, i, j, v] for t2 in T if t2 <= t for j in R),
            "blockRelayOptionallyOnlyIfEnoughCommits(%s,%s,%s)" % (t, i, v),
        )

    for (i, v) in product(R, V):
        m += (
            xsum(SendPrepReq[t, i, v] for t in T - {1}) <= Primary[i, v],
            "prepReqOOIfPrimary(%s,%s)" % (i, v),
        )
        m += (
            xsum(SendPrepRes[t, i, v] for t in T - {1}) <= 1,
            "sendPrepResOO(%s,%s)" % (i, v),
        )
        m += (
            xsum(SendCommit[t, i, v] for t in T - {1}) <= 1,
            "sendCommitOO(%s,%s)" % (i, v),
        )
        m += (
            xsum(SendCV[t, i, v] for t in T - {1}) <= 1,
            "sendCVOO(%s,%s)" % (i, v),
        )
        m += (
            xsum(BlockRelay[t, i, v] for t in T - {1}) <= 1,
            "blockRelayOO(%s,%s)" % (i, v),
        )

    for (t, i, v) in product(T - {1}, R, V):
        m += (
            RecvPrepReq[t, i, i, v] == SendPrepReq[t, i, v],
            "prepReqReceivedSS(%s,%s,%s)" % (t, i, v),
        )
        m += (
            RecvPrepResp[t, i, i, v] == SendPrepRes[t, i, v],
            "prepRespReceivedSS(%s,%s,%s)" % (t, i, v),
        )
        m += (
            RecvCommit[t, i, i, v] == SendCommit[t, i, v],
            "commitReceivedSS(%s,%s,%s)" % (t, i, v),
        )
        m += (
            RecvCV[t, i, i, v] == SendCV[t, i, v],
            "cvReceivedSS(%s,%s,%s)" % (t, i, v),
        )

    for t, i, j, v in product(T - {1}, R, R, V):
        if i != j:
            m += (
                RecvPrepReq[t, i, j, v]
                <= xsum(SendPrepReq[t2, j, v] for t2 in T if 1 < t2 < t),
                "prepReqReceived(%s,%s,%s,%s)" % (t, i, j, v),
            )
            m += (
                RecvPrepResp[t, i, j, v]
                <= xsum(SendPrepRes[t2, j, v] for t2 in T if 1 < t2 < t),
                "prepRespReceived(%s,%s,%s,%s)" % (t, i, j, v),
            )
            m += (
                RecvCommit[t, i, j, v]
                <= xsum(SendCommit[t2, j, v] for t2 in T if 1 < t2 < t),
                "commitReceived(%s,%s,%s,%s)" % (t, i, j, v),
            )
            m += (
                RecvCV[t, i, j, v] <= xsum(SendCV[t2, j, v] for t2 in T if 1 < t2 < t),
                "cvReceived(%s,%s,%s,%s)" % (t, i, j, v),
            )

    for (t, i, j, v) in product(T - {1}, R, R, V):
        m += (
            RecvPrepResp[t, i, j, v] >= RecvPrepReq[t, i, j, v],
            "prepResReceivedAlongWithPrepReq(%s,%s,%s,%s)" % (t, i, j, v),
        )

    for (i, j, v) in product(R, R, V):
        m += (
            xsum(RecvPrepReq[t, i, j, v] for t in T - {1}) <= 1,
            "rcvdPrepReqOO(%s,%s,%s)" % (i, j, v),
        )
        m += (
            xsum(RecvPrepResp[t, i, j, v] for t in T - {1}) <= 1,
            "rcvdPrepResOO(%s,%s,%s)" % (i, j, v),
        )
        m += (
            xsum(RecvCommit[t, i, j, v] for t in T - {1}) <= 1,
            "rcvdCommitOO(%s,%s,%s)" % (i, j, v),
        )
        m += (
            xsum(RecvCV[t, i, j, v] for t in T - {1}) <= 1,
            "rcvdCVOO(%s,%s,%s)" % (i, j, v),
        )

    for v in V:
        m += (
            blockRelayed[v] <= xsum(BlockRelay[t, i, v] for t in T - {1} for i in R),
            "blockRelayedOnlyIfNodeRelay(%s)" % (v),
        )
        m += (
            blockRelayed[v] * N >= xsum(BlockRelay[t, i, v] for t in T - {1} for i in R),
            "blockRelayedCounterForced(%s)" % (v),
        )

    for (i, j, v) in product(R_OK, R_OK, V):
        if i != j:
            m += (
                xsum(RecvCV[t, i, j, v] for t in T - {1})
                >= xsum(SendCV[t, j, v] for t in T - {1}),
                "cvReceivedNonByz(%s,%s,%s)" % (i, j, v),
            )

    for i in R_OK:
        m += (
            xsum(BlockRelay[t, i, v] for t in T - {1} for v in V) <= 1,
            "blockRelayLimitToOneForNonByz(%s)" % (i),
        )
    for (i, v) in product(R_OK, V - {1}):
        m += (
            xsum(2 * Primary[ii, v] for ii in R)
            >= changeViewRecvPerNodeAndView[i, v - 1] - M + 1,
            "assertAtLeastOnePrimaryIfEnoughCV(%s,%s)" % (i, v),
        )
    for (i, v) in product(R_OK, V):
        m += (
            xsum(SendPrepReq[t, i, v] for t in T - {1}) >= Primary[i, v],
            "assertSendPrepReqWithinSimLimit(%s,%s)" % (i, v),
        )
        m += (
            xsum(SendPrepRes[t, i, v] for t in T - {1})
            >= xsum(RecvPrepReq[t, i, j, v] for (t, j) in product(T - {1}, R)),
            "assertSendPrepResWithinSimLimit(%s,%s)" % (i, v),
        )
        m += (
            xsum(2 * SendCommit[t, i, v] for t in T - {1})
            >= xsum(RecvPrepResp[t, i, j, v] for (t, j) in product(T - {1}, R)) - M + 1,
            "assertSendCommitWithinSimLimit(%s,%s)" % (i, v),
        )
        m += (
            xsum(2 * BlockRelay[t, i, v] for t in T - {1})
            >= xsum(RecvCommit[t, i, j, v] for (t, j) in product(T - {1}, R)) - M + 1,
            "assertBlockRelayWithinSimLimit(%s,%s)" % (i, v),
        )

    for (i, v) in product(R_OK, V - {1}):
        m += (
            xsum(SendPrepReq[t, i, v] for t in T - {1})
            <= (1 / M) * changeViewRecvPerNodeAndView[i, v - 1],
            "sendPrepReqOnlyIfViewBeforeOk(%s,%s)" % (i, v),
        )
        m += (
            xsum(SendPrepRes[t, i, v] for t in T - {1})
            <= (1 / M) * changeViewRecvPerNodeAndView[i, v - 1],
            "sendPrepResOnlyIfViewBeforeOk(%s,%s)" % (i, v),
        )
        m += (
            xsum(SendCommit[t, i, v] for t in T - {1})
            <= (1 / M) * changeViewRecvPerNodeAndView[i, v - 1],
            "sendCommitOnlyIfViewBeforeOk(%s,%s)" % (i, v),
        )
        m += (
            xsum(SendCV[t, i, v] for t in T - {1})
            <= (1 / M) * changeViewRecvPerNodeAndView[i, v - 1],
            "sendCVOnlyIfViewBeforeOk(%s,%s)" % (i, v),
        )

    for i in R_OK:
        m += (
            xsum(SendCV[t, i, 1] for t in T - {1})
            >= 1 - xsum(SendCommit[t, i, 1] for t in T - {1}),
            "assertSendCVIfNotSendCommitV1(%s)" % (i),
        )

    for (i, v) in product(R_OK, V - {1}):
        m += (
            xsum(SendCV[t, i, v] for t in T - {1})
            >= 1
            - xsum(SendCommit[t, i, v] for t in T - {1})
            - (1 - xsum(Primary[ii, v - 1] for ii in R)),
            "assertSendCVIfNotCommitAndYesPrimary(%s,%s)" % (i, v),
        )

    for (i, v, t) in product(R_OK, V, T - {1}):
        m += (
            SendPrepReq[t, i, v]
            <= 1 - xsum(SendCV[t2, i, v] for t2 in T if 1 < t2 <= t),
            "noPrepReqIfCV(%s,%s,%s)" % (i, v, t),
        )
        m += (
            SendPrepRes[t, i, v]
            <= 1 - xsum(SendCV[t2, i, v] for t2 in T if 1 < t2 <= t),
            "noPrepResIfCV(%s,%s,%s)" % (i, v, t),
        )
        m += (
            SendCommit[t, i, v] <= 1 - xsum(SendCV[t2, i, v] for t2 in T if 1 < t2 <= t),
            "noCommitIfCV(%s,%s,%s)" % (i, v, t),
        )
        m += (
            SendCV[t, i, v] <= 1 - xsum(SendCommit[t2, i, v] for t2 in T if 1 < t2 <= t),
            "noCVIfCommit(%s,%s,%s)" % (i, v, t),
        )
        m += (
            SendPrepReq[t, i, v]
            <= 1 - xsum(BlockRelay[t2, i, v] for t2 in T if 1 < t2 <= t),
            "noBlockYesPrepReq(%s,%s,%s)" % (i, v, t),
        )
        m += (
            SendPrepRes[t, i, v]
            <= 1 - xsum(BlockRelay[t2, i, v] for t2 in T if 1 < t2 <= t),
            "noBlockYesPrepRes(%s,%s,%s)" % (i, v, t),
        )
        m += (
            SendCommit[t, i, v]
            <= 1 - xsum(BlockRelay[t2, i, v] for t2 in T if 1 < t2 <= t),
            "noBlockYesCommit(%s,%s,%s)" % (i, v, t),
        )
        m += (
            SendCV[t, i, v] <= 1 - xsum(BlockRelay[t2, i, v] for t2 in T if 1 < t2 <= t),
            "noBlockYesCV(%s,%s,%s)" % (i, v, t),
        )

    for (i, v) in product(R_OK, V - {1}):
        m += (
            Primary[i, v]
            <= 1 - xsum(BlockRelay[t, i, v2] for t in T - {1} for v2 in V if v2 < v),
            "noBlockOldViewsYesPrimary(%s,%s)" % (i, v),
        )
        m += (
            xsum(SendPrepReq[t, i, v] for t in T - {1})
            <= 1 - xsum(BlockRelay[t, i, v2] for t in T - {1} for v2 in V if v2 < v),
            "noBlockOldViewsYesPrepReq(%s,%s)" % (i, v),
        )
        m += (
            xsum(SendPrepRes[t, i, v] for t in T - {1})
            <= 1 - xsum(BlockRelay[t, i, v2] for t in T - {1} for v2 in V if v2 < v),
            "noBlockOldViewsYesPrepRes(%s,%s)" % (i, v),
        )
        m += (
            xsum(SendCommit[t, i, v] for t in T - {1})
            <= 1 - xsum(BlockRelay[t, i, v2] for t in T - {1} for v2 in V if v2 < v),
            "noBlockOldViewsYesCommit(%s,%s)" % (i, v),
        )
        m += (
            xsum(SendPrepReq[t, i, v] for t in T - {1})
            <= 1 - xsum(SendCommit[t, i, v2] for t in T - {1} for v2 in V if v2 < v),
            "noCommitOldViewsYesPrepReq(%s,%s)" % (i, v),
        )
        m += (
            xsum(SendPrepRes[t, i, v] for t in T - {1})
            <= 1 - xsum(SendCommit[t, i, v2] for t in T - {1} for v2 in V if v2 < v),
            "noCommitOldViewsYesPrepRes(%s,%s)" % (i, v),
        )
        m += (
            xsum(SendCommit[t, i, v] for t in T - {1})
            <= 1 - xsum(SendCommit[t, i, v2] for t in T - {1} for v2 in V if v2 < v),
            "noCommitOldViewsYesCommit(%s,%s)" % (i, v),
        )
        m += (
            xsum(SendCV[t, i, v] for t in T - {1})
            <= 1 - xsum(SendCommit[t, i, v2] for t in T - {1} for v2 in V if v2 < v),
            "noCommitOldViewsYesCV(%s,%s)" % (i, v),
        )

    m += (
        totalBlockRelayed == xsum(blockRelayed[v] for v in V),
        "calcSumBlockRelayed",
    )

    m += (
        numberOfRounds == xsum(Primary[i, v] for (i, v) in product(R, V)),
        "calcTotalPrimaries",
    )

    for (i, v) in product(R, V):
        m += (
            changeViewRecvPerNodeAndView[i, v]
            == xsum(RecvCV[t, i, j, v] for (t, j) in product(T, R)),
            "calcChangeViewEveryNodeAndView(%s,%s)" % (i, v),
        )

    for (t, i, v) in product(T, R, V):
        m += (
            lastRelayedBlock
            >= ((v - 1) * tMax * BlockRelay[t, i, v] + BlockRelay[t, i, v] * t),
            "calcLastRelayedBlockMaxProblem(%s,%s,%s)" % (t, i, v),
        )

    m.objective = maximize(totalBlockRelayed * 1000 + numberOfRounds)

    return m


@pytest.mark.parametrize("pdata", PDATA.keys())
@pytest.mark.parametrize("solver", SOLVERS)
def test_dbft_mip(solver, pdata):
    """run tests"""
    N, tMax = pdata
    m = create_model(solver, N, tMax)
    (opt, cols, rows, nzs) = PDATA[pdata]
    assert m.num_cols == cols
    assert m.num_rows == rows
    assert m.num_nz == nzs

    if cols + rows > 50000:
        return

    m.verbose = 0
    m.max_nodes = 50
    m.optimize()
    assert m.objective_bound >= opt - TOL
    if m.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
        m.check_optimization_results()
        if m.status == OptimizationStatus.OPTIMAL:
            assert abs(m.objective_value - opt) <= TOL
        assert abs(m.objective.x - m.objective_value) <= TOL
