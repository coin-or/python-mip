from itertools import product
from sys import stdout as out
from mip import Model, xsum, minimize, BINARY
from mip.callbacks import ConstrsGenerator, CutPool
from typing import Set, List
from collections import defaultdict

def subtour(N: Set, out: defaultdict, node) -> List:
    """verifica se 'node' pertence a uma sub-rota. 
    se positivo retorna os elementos dessa sub-rota"""
    queue = [node]
    visited = set(queue)
    while queue:
        n = queue.pop()
        for nl in out[n]:
            if nl not in visited:
                queue.append(nl)
                visited.add(nl)

    if len(visited) != len(N):
        return [v for v in visited]
    else:
        return []

class SubTourLazyGenerator(ConstrsGenerator):
    """Gera restrições de eliminação de sub-rotas em soluções inteiras"""
    def generate_constrs(self, model: Model):
        r = [(v, v.x) for v in model.vars
             if v.name.startswith('x(') and v.x >= 0.99]
        U = [int(v.name.split('(')[1].split(',')[0]) for v, f in r]
        V = [int(v.name.split(')')[0].split(',')[1]) for v, f in r]
        N, cp = set(U+V), CutPool()
        # output nodes for each node
        out = defaultdict(lambda: list())
        for i in range(len(U)):
            out[U[i]].append(V[i])

        for n in N:
            S = set(subtour(N, out, n))
            if S:
                arcsInS = [(v, f) for i, (v, f) in enumerate(r)
                           if U[i] in S and V[i] in S]
                if sum(f for v, f in arcsInS) >= (len(S)-1)+1e-4:
                    cut = xsum(1.0*v for v, fm in arcsInS) <= len(S)-1
                    cp.add(cut)
        for cut in cp.cuts:
            model += cut


# locais a visitar
places = ['Antwerp', 'Bruges', 'C-Mine', 'Dinant', 'Ghent',
          'Grand-Place de Bruxelles', 'Hasselt', 'Leuven',
          'Mechelen', 'Mons', 'Montagne de Bueren', 'Namur',
          'Remouchamps', 'Waterloo']

# matriz triangular superior com tempos de deslocamento
dists = [[83, 81, 113, 52, 42, 73, 44, 23, 91, 105, 90, 124, 57],
         [161, 160, 39, 89, 151, 110, 90, 99, 177, 143, 193, 100],
         [90, 125, 82, 13, 57, 71, 123, 38, 72, 59, 82],
         [123, 77, 81, 71, 91, 72, 64, 24, 62, 63],
         [51, 114, 72, 54, 69, 139, 105, 155, 62],
         [70, 25, 22, 52, 90, 56, 105, 16],
         [45, 61, 111, 36, 61, 57, 70],
         [23, 71, 67, 48, 85, 29],
         [74, 89, 69, 107, 36],
         [117, 65, 125, 43],
         [54, 22, 84],
         [60, 44],
         [97],
         []]

# nr. de pontos e conjunto de pontos
n, V = len(dists), set(range(len(dists)))

# matriz com tempos
c = [[0 if i == j
      else dists[i][j-i-1] if j > i
      else dists[j][i-j-1]
      for j in V] for i in V]

model = Model()

# variáveis 0/1 indicando se um arco (i,j) participa da rota ou não
x = [[model.add_var(var_type=BINARY) for j in V] for i in V]

# variáveis auxiliares
y = [model.add_var() for i in V]

# função objetivo: minimizar tempo
model.objective = minimize(xsum(c[i][j]*x[i][j] for i in V for j in V))

# restrição: selecionar arco de saída da cidade
for i in V:
    model += xsum(x[i][j] for j in V - {i}) == 1

# restrição: selecionar arco de entrada na cidade
for i in V:
    model += xsum(x[j][i] for j in V - {i}) == 1

# eliminação de sub-rotas
for (i, j) in product(V - {0}, V - {0}):
    model += y[i] - (n+1)*x[i][j] >= y[j]-n

# chamada da otimização com limite tempo de 30 segundos
model.lazy_constrs_generator = SubTourLazyGenerator()
model.optimize(max_seconds=30)

# verificando se ao menos uma solução foi encontrada e a imprimindo
if model.num_solutions:
    out.write('route with total distance %g found: %s'
              % (model.objective_value, places[0]))
    nc = 0
    while True:
        nc = [i for i in V if x[nc][i].x >= 0.99][0]
        out.write(' -> %s' % places[nc])
        if nc == 0:
            break
    out.write('\n')
