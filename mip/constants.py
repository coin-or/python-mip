"""Python-MIP constants"""

from enum import Enum

VERSION = "1.6.8"

# epsilon number (practical zero)
EPS = 10e-64

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
CPX = "CPX"  # we plan to support CPLEX in the future
CPLEX = "CPX"  # we plan to support CPLEX in the future
GRB = "GRB"
GUROBI = "GRB"
SCIP = "SCIP"  # we plan to support SCIP in the future

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


class LP_Method(Enum):
    """Different methods to solve the linear programming problem."""

    AUTO = 0
    """Let the solver decide which is the best method"""

    DUAL = 1
    """The dual simplex algorithm"""

    PRIMAL = 2
    """The primal simplex algorithm"""

    BARRIER = 3
    """The barrier algorithm"""
