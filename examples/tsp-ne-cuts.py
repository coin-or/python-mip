"""Non-euclidean TSP cutting plane example"""
from mip.model import *
from itertools import product
from collections import defaultdict
from networkx import minimum_cut,DiGraph
N = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

A = { ('a','d') : 56, ('d','a') : 67, ('a','b') : 49, ('b','a') : 50,
      ('d','b') : 39, ('b','d') : 37, ('c','f') : 35, ('f','c') : 35, 
      ('g','b') : 35, ('b','g') : 35, ('g','b') : 35, ('b','g') : 25, 
      ('a','c') : 80, ('c','a') : 99, ('e','f') : 20, ('f','e') : 20, 
      ('g','e') : 38, ('e','g') : 49, ('g','f') : 37, ('f','g') : 32, 
      ('b','e') : 21, ('e','b') : 30, ('a','g') : 47, ('g','a') : 68,
      ('d','c') : 37, ('c','d') : 52, ('d','e') : 15, ('e','d') : 20 }

# input and output arcs per node
Aout = {n:[a for a in A if a[0]==n] for n in N}
Ain  = {n:[a for a in A if a[1]==n] for n in N}

m=Model(solver_name='cbc')
m.verbose=0

x = {a:m.add_var(name='x({},{})'.format(a[0], a[1]), var_type=BINARY) for a in A}
m.objective = xsum(c*x[a] for a,c in A.items())

for n in N:
    m += xsum(x[a] for a in Aout[n]) == 1, 'out({})'.format(n)
    m += xsum(x[a] for a in Ain[n]) == 1, 'in({})'.format(n)
 
newConstraints=True
while newConstraints:
    m.relax()
    m.optimize()
    print('objective value : {}'.format(m.objective_value))
    G = DiGraph()
    for a in A:
        G.add_edge(a[0], a[1], capacity=x[a].x);
    newConstraints=False
    for (n1,n2) in [(i,j) for (i,j) in product(N,N) if i!=j]:
        cut_value, (S,NS) = minimum_cut(G, n1, n2)
        if (cut_value<=0.99):
            m += xsum(x[a] for a in A if (a[0] in S and a[1] in S)) <= len(S)-1
            newConstraints = True

        
