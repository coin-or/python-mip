'''
Reads instance data for the bandwidth multicoloring problem
in the .col format

Created on May 20, 2019

@author: haroldo
'''

from collections import defaultdict
from typing import Tuple, List, DefaultDict


class BMCPData:
    def __init__(self,
                 N: List[int],
                 r: List[int],
                 d: List[DefaultDict[int, int]]):
        self.N = N  # node list
        self.r = r  # required channels per node
        self.d = d  # distances matrix, d[i][j] indicates the minimum
                    # distance between nodes i and j


def read(file_name: str) -> \
        BMCPData:
    """reads a Bandwidth Multicoloring Problem instance
    """
    n, m = 0, 0
    f = open(file_name, 'r')
    for line in f:
        line = line.rstrip().lstrip().lower()
        line = ' '.join(line.split())
        lc = line.split(' ')

        if lc[0] == 'p':
            n, m = int(lc[2]), int(lc[3])
            d = [defaultdict(int) for i in range(n)]
            r = [int(1) for i in range(n)]
        elif lc[0] == 'e':
            u, v, w = int(lc[1])-1, int(lc[2])-1, int(lc[3])
            d[u][v] = w
            d[v][u] = w
        elif lc[0] == 'n':
            r[int(lc[1])-1] = int(lc[2])

    f.close()
    N = [i for i in range(n)]
    data = BMCPData(N, r, d)
    return data
