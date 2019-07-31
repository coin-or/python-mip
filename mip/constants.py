"""Python-MIP constants"""

from enum import Enum

VERSION = '1.3.6'

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
    """ Status of the optimization """

    ERROR = -1
    """ Solver returned an error"""

    OPTIMAL = 0
    """Optimal solution was computed"""

    INFEASIBLE = 1
    """The model is proven infeasible"""

    UNBOUNDED = 2
    """One or more variables that appear in the objective function are not
       included in binding constraints and the optimal objective value is
       infinity."""

    FEASIBLE = 3
    """An integer feasible solution was found during the search but the search
       was interrupted before concluding if this is the optimal solution or
       not."""

    INT_INFEASIBLE = 4
    """A feasible solution exist for the relaxed linear program but not for the
       problem with existing integer variables"""

    NO_SOLUTION_FOUND = 5
    """A truncated search was executed and no integer feasible solution was
    found"""

    LOADED = 6
    """The problem was loaded but no optimization was performed"""

    CUTOFF = 7
    """No feasible solution exists for the current cutoff"""

    OTHER = 10000


# search emphasis
class SearchEmphasis(Enum):
    DEFAULT = 0
    FEASIBILITY = 1
    OPTIMALITY = 2
