'''
Created on May 20, 2019

@author: haroldo
'''

from bmcp_data import BMCPData
from bmcp_sol import BMCPSol

def build(data: BMCPData) -> BMCPSol:
    S = BMCPSol(data)
    N, r, d = data.N, data.r, data.d

    # list of nodes sorted by conflicts
    L = [(sum(1 for v in d[i].values()), i)
            for i in N]
    L.sort(reverse=True)
    for (v, u) in L:
        print('allocating node {} which has {} conflicts'.format(u, v))

        node = u
        for i in range(r[node]):
            av = S.available_color(u)
            S.allocate(node, av)

    return S


