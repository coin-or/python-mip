from math import inf, ceil, cos, acos, floor, radians
from typing import List

# constants as stated in TSPlib doc
# https://www.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/tsp95.pdf
PI = 3.141592
RRR = 6378.388

# convert to radians
def rad(val : float) -> float:
    mult = 1.0
    if val<0.0:
        mult = -1.0
        val = abs(val)
    
    deg = float(floor(val))
    minute = val - deg
    return (PI * (deg + 5*minute/3)/180)*mult

class TSPData:
    """Reads instances from the Traveling Salesman Problem
    
    Right now only supports instances containing geographical coordinates 
    
    Attributes:
    
        n (int): number of cities
        d (List[List[int]]): distance matrix
    
    """
    def __init__(self, fileName : str):
        self.name = ''
        
        self.n = 0 
        
        self.d = None
        
        self.latitude = []
        
        self.longitude = []
        
        readingCoord = False
        
        self.x  = []
        self.y  = []
        
        self.ix = []
        self.iy = []
        
        f=open(fileName, 'r')
        for l in f:
            if ':' in l:
                vls = l.split(':')
                fn = vls[0].lower()
                fv = vls[1]
                if fn == 'name':
                    self.name = fv.strip()
                elif fn == 'dimension':
                    self.n = int(fv.strip())
                    self.d = [[inf for i in range(self.n)] for j in range(self.n)]
                    self.latitude = [float(0) for i in range(self.n)]
                    self.longitude = [float(0) for i in range(self.n)]
                    self.x = [float(0) for i in range(self.n)]
                    self.y = [float(0) for i in range(self.n)]
                    self.ix = [int(0) for i in range(self.n)]
                    self.iy = [int(0) for i in range(self.n)]
                    
                    self.names = ['city{}'.format(i+1) for i in range(self.n)]
            elif 'NODE_COORD_SECTION' in l:
                readingCoord = True
            elif readingCoord:
                l = l.lstrip().rstrip().lower()
                if 'eof' in l:
                    break                
                vls = l.split(' ')
                i = int(vls[0])
                cx = float(vls[1])
                cy = float(vls[2])
                    
                self.x[i-1] = cx 
                self.y[i-1] = cy

                if len(vls)>3:
                    self.ix[i-1] = int(vls[3]) 
                    self.iy[i-1] = int(vls[4])
                    #print('i {} {}'.format(self.ix[-1], self.iy[-1]))
                    if len(vls)>5:
                        self.names[i-1] = ''
                        for j in range(5, len(vls)):
                            self.names[i-1] += vls[j]
                        
                        
                        

        for i in range(self.n):
            self.latitude[i] = radians(self.x[i])
            self.longitude[i] = radians(self.y[i])

        for i in range(self.n):
            self.d[i][i] = 0
            for j in range(0, self.n):
                q1 = cos( self.longitude[i] - self.longitude[j] )
                q2 = cos( self.latitude[i] - self.latitude[j] )
                q3 = cos( self.latitude[i] + self.latitude[j] )
                self.d[i][j] = int(floor(RRR*acos(0.5*((1.0+q1)*q2-(1.0-q1)*q3))+1.0)) 
                
        """
        ff = open('t.data', 'w')
        ff.write('data;\n\n')
        ff.write('set V :=')
        for i in range(self.n):
            ff.write(' {}'.format(i+1))
        ff.write('\n\n')
        ff.write('param : A : c :=\n')
        for i in range(self.n):
            for j in range(self.n):
                if i==j:
                    continue
                ff.write('\t{} {} {}\n'.format(i+1,j+1, self.d[i][j]))
        ff.write(';\n\nend;\n\n')

        ff.close()
        """
