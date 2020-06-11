"""Python-MIP constants"""

from enum import Enum

VERSION = "1.9.2"

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

# cutting planes types
class CutType(Enum):
    """ Types of cuts that can be generated"""

    GOMORY = 0
    """Gomory Mixed Integer cuts [Gomo69]_ ."""

    MIR = 1
    """Mixed-Integer Rounding cuts [Marc01]_."""

    ZERO_HALF = 2
    """Zero/Half cuts [Capr96]_."""

    CLIQUE = 3
    """Clique cuts [Padb73]_."""

    KNAPSACK_COVER = 4
    """Knapsack cover cuts [Bala75]_."""

    LIFT_AND_PROJECT = 5
    """Lift-and-project cuts [BCC93]_."""


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
    """Default search emphasis, try to balance between improving the dual
    bound and producing integer feasible solutions."""

    FEASIBILITY = 1
    """More aggressive search for feasible solutions."""

    OPTIMALITY = 2
    """Focuses more on producing improved dual bounds even if the
    production of integer feasible solutions is delayed."""


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
