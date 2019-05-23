'''
Created on May 20, 2019

@author: haroldo
'''

from collections import defaultdict
from typing import List, DefaultDict, Tuple
from itertools import product
from bmcp_data import BMCPData

class BMCPSol():
    """
    Bandwidth Multi Coloring Problem
    """

    def __init__(self,
                 data: BMCPData):  # distance adjacency lists
        # instance data
        self.data = data

        # solution
        self.C = [[] for i in self.data.N]
        self.u_max = 0

    def allocate(self, i: int, c: int):
        """allocated color c to node i"""
        self.C[i].append(c)
        self.u_max = max(self.u_max, c)

    def available_color(self, i: int) -> int:
        """ returns the next available color for node i """
        ac = 0
        available = False
        has_conflict = False
        d, C = self.data.d, self.C
        while available is False:
            for (v, dist) in d[i].items():
                for c in C[v]:
                    if abs(ac-c) < dist:
                        available = False
                        ac += 1
                        has_conflict = True
                        break
                if has_conflict:
                    break
            if has_conflict:
                has_conflict = False
                continue

            available = True

        return ac

    def __str__(self) -> str:
        N, C, u_max = self.data.N, self.C, self.u_max
        result = '{} different colors used in the solution:\n'.format(u_max+1)
        for node in N:
            result += '\t[{}]'.format(node+1)
            for color in C[node]:
                result += ' {}'.format(color+1)
            result += '\n'

        return result
