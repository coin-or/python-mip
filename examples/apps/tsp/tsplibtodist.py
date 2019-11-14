#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 haroldo <haroldo@viper>
#
# Distributed under terms of the MIT license.

from sys import argv
from os.path import basename
import tsplib95 as tsp_data

P = tsp_data.load_problem(argv[1])

iname = basename(argv[1])

if iname.endswith('.tsp'):
    iname = iname.split('.tsp')[0]

N = set([i for i in P.get_nodes()])

fo = open('%s.dist' % iname, 'w')
fo.write('%d\n' % len(P))
for i in N:
    for j in N:
        fo.write('%d\n' % P.wfunc(i, j))
fo.close()
