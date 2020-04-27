#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 haroldo <haroldo@soyuz>
#
# Distributed under terms of the MIT license.


"""Creates a compact formulation for the two dimensional bin packing problem"""

from mip import Model, BINARY, CONTINUOUS, INTEGER, xsum, minimize, MINIMIZE


def create_mip(solver, w, h, W, relax=False):
    m = Model(solver_name=solver)
    n = len(w)
    I = set(range(n))
    S = [[j for j in I if h[j] <= h[i]] for i in I]
    G = [[j for j in I if h[j] >= h[i]] for i in I]

    if relax:
        x = [
            {
                j: m.add_var(
                    var_type=CONTINUOUS,
                    lb=0.0,
                    ub=1.0,
                    name="x({},{})".format(i, j),
                )
                for j in S[i]
            }
            for i in I
        ]
    else:
        x = [
            {
                j: m.add_var(var_type=BINARY, name="x({},{})".format(i, j))
                for j in S[i]
            }
            for i in I
        ]

    if relax:
        vtoth = m.add_var(name="H", lb=0.0, ub=sum(h), var_type=CONTINUOUS)
    else:
        vtoth = m.add_var(name="H", lb=0.0, ub=sum(h), var_type=INTEGER)

    toth = xsum(h[i] * x[i][i] for i in I)

    m.objective = minimize(toth)

    # each item should appear as larger item of the level
    # or as an item which belongs to the level of another item
    for i in I:
        m += xsum(x[j][i] for j in G[i]) == 1, "cons(1,{})".format(i)

    # represented items should respect remaining width
    for i in I:
        m += (
            (
                xsum(w[j] * x[i][j] for j in S[i] if j != i)
                <= (W - w[i]) * x[i][i]
            ),
            "cons(2,{})".format(i),
        )

    return m
