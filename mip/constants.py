VERSION = '1.0.26'

# epsilon number (practical zero)
EPS = 10e-6

# infinity representation
INF = float("inf")

# optimization status
ERROR = -1
OPTIMAL = 0
INFEASIBLE = 1
UNBOUNDED = 2
FEASIBLE = 3
INT_INFEASIBLE = 4
NO_SOLUTION_FOUND = 5
LOADED = 6
CUTOFF = 7

# constraint senses
EQUAL = "="
LESS_OR_EQUAL = "<"
GREATER_OR_EQUAL = ">"

# optimization directions
MINIMIZE = "MIN"
MAXIMIZE = "MAX"

# Search emphasis
FEASIBILITY = 1
OPTIMALITY = 2

# variable types
BINARY = "B"
CONTINUOUS = "C"
INTEGER = "I"

# solvers
CBC = "CBC"
CPLEX = "CPX"
GUROBI = "GRB"
SCIP = "SCIP"
