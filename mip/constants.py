VERSION = '1.1.1'

from enum import Enum

# epsilon number (practical zero)
EPS = 10e-6

# infinity representation
INF = float("inf")


# constraint senses
EQUAL = "="
LESS_OR_EQUAL = "<"
GREATER_OR_EQUAL = ">"

# optimization directions
MIN = "MIN"
MAX = "MAX"
MINIMIZE = "MIN"
MAXIMIZE = "MAX"

# solvers
CBC = "CBC"
CPX = "CPX"
CPLEX = "CPX"
GRB = "GRB"
GUROBI = "GRB"
SCIP = "SCIP"

# variable types
BINARY = "B"
CONTINUOUS = "C"
INTEGER = "I"


# optimization status
class OptimizationStatus(Enum):
    ERROR = -1
    OPTIMAL = 0
    INFEASIBLE = 1
    UNBOUNDED = 2
    FEASIBLE = 3
    INT_INFEASIBLE = 4
    NO_SOLUTION_FOUND = 5
    LOADED = 6
    CUTOFF = 7
    OTHER = 10000


# search emphasis
class SearchEmphasis(Enum):
    DEFAULT = 0
    FEASIBILITY = 1
    OPTIMALITY = 2
