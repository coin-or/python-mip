#JSSPInstance.py
class JSSPInstance:
    def __init__(self, instanceName):
        self.instanceName = instanceName
        with open(instanceName,"r") as instance:
            line = instance.readline()
            self.n, self.m = (int(x) for x in line.split())
            self.machines = [[0]*self.m for i in range(self.n)]
            self.times = [[0]*self.m for i in range(self.n)]
            self.M = 0
            
            for j in range(self.n):
                line = instance.readline()
                line = line.split()
                i = 0
                while i < self.m:
                    value = int(line[2*i])
                    self.machines[j][i] = value
                    value = int(line[2*i+1])
                    self.times[j][int(line[2*i])] = value
                    self.M += value
                    i = i+1
