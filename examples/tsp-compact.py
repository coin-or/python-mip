from tspdata import TSPData
from sys import argv
from mip.model import Model
from mip.constants import *
from matplotlib.pyplot import plot

if len(argv) <= 1:
    print('enter instance name.')
    exit(1)
    
inst = TSPData(argv[1])
n = inst.n
d = inst.d
print('solving TSP with {} cities'.format(inst.n))

model = Model()

# binary variables indicating if arc (i,j) is used on the route or not
x = [ [ model.add_var(
           name='x({},{})'.format(i,j), 
           type=BINARY) 
             for j in range(n) ] 
               for i in range(n) ]

# continuous variable to prevent subtours: each
# city will have a different "identifier" in the planned route
y = [ model.add_var(
       name='y({})'.format(i),
       lb=0.0,
       ub=n) 
         for i in range(n) ]

# objective funtion: minimize the distance
model += sum( d[i][j]*x[i][j]
                for j in range(n) for i in range(n) )

# constraint : enter each city coming from another city
for i in range(n):
    model += sum( x[j][i] for j in range(n) if j != i ) == 1, 'enter({})'.format(i)
    
# constraint : leave each city coming from another city
for i in range(n):
    model += sum( x[i][j] for j in range(n) if j != i ) == 1, 'leave({})'.format(i)
    
# subtour elimination
for i in range(0, n):
    for j in range(0, n):
        if i==j or i==0 or j==0:
            continue
        model += \
            y[i] - y[j] - (n+1)*x[i][j] >= -n, 'noSub({},{})'.format(i,j)
                 
    
model.optimize( maxSeconds=10 )
#model.write('tsp.lp')

print('best route found has length {}'.format(model.get_objective_value()))

for i in range(n):
    for j in range(n):
        if x[i][j].x >= 0.98:
            print('arc ({},{})'.format(i,j))

print('finished')        
    

