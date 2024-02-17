"""Python-MIP constants"""

from enum import Enum
from cffi import FFI

ffi = FFI()

# epsilon number (practical zero)
EPS = 10e-64

# infinity representation
INF = float("inf")
INT_MAX = 2 ** (ffi.sizeof("int") * 8 - 2)

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
HIGHS = "HiGHS"
SCIP = "SCIP"  # we plan to support SCIP in the future

# variable types
BINARY = "B"
CONTINUOUS = "C"
INTEGER = "I"

# cutting planes types
class CutType(Enum):

    """Types of cuts that can be generated. Each cut type is an implementation
    in the `COIN-OR Cut Generation Library <https://github.com/coin-or/Cgl>`_.
    For some cut types multiple implementations are available. Sometimes these
    implementations were designed with different objectives: for the generation
    of Gomory cutting planes, for example, the GMI cuts are focused on numerical
    stability, while Forrest's implementation (GOMORY) is more integrated into
    the CBC code."""

    PROBING = 0
    """Cuts generated evaluating the impact of fixing bounds for integer
    variables"""

    GOMORY = 1
    """Gomory Mixed Integer cuts [Gomo69]_, as implemented by John Forrest."""

    GMI = 2
    """Gomory Mixed Integer cuts [Gomo69]_, as implemented by Giacomo
    Nannicini, focusing on numerically safer cuts."""

    RED_SPLIT = 3
    """Reduce and split cuts [AGY05]_, implemented by Francois Margot."""

    RED_SPLIT_G = 4
    """Reduce and split cuts [AGY05]_, implemented by Giacomo Nannicini."""

    FLOW_COVER = 5
    """Lifted Simple Generalized Flow Cover Cut Generator."""

    MIR = 6
    """Mixed-Integer Rounding cuts [Marc01]_."""

    TWO_MIR = 7
    """Two-phase Mixed-integer rounding cuts."""

    LATWO_MIR = 8
    """Lagrangean relaxation for two-phase Mixed-integer rounding cuts, as in
    LAGomory"""

    LIFT_AND_PROJECT = 9
    """Lift-and-project cuts [BCC93]_, implemented by Pierre Bonami."""

    RESIDUAL_CAPACITY = 10
    """Residual capacity cuts [AtRa02]_, implemented by Francisco Barahona."""

    ZERO_HALF = 11
    """Zero/Half cuts [Capr96]_."""

    CLIQUE = 12
    """Clique cuts [Padb73]_."""

    ODD_WHEEL = 13
    """Lifted odd-hole inequalities."""

    KNAPSACK_COVER = 14
    """Knapsack cover cuts [Bala75]_."""


# optimization status
class OptimizationStatus(Enum):
    """Status of the optimization"""

    ERROR = -1
    """Solver returned an error"""

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

    INF_OR_UNBD = 8
    """Special state for gurobi solver. In some cases gurobi could not 
    determine if the problem is infeasible or unbounded due to application
    of dual reductions (when active) during presolve."""

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

    BARRIERNOCROSS = 4
    """The barrier algorithm without performing crossover"""


class ConstraintPriority(Enum):
    """A constraint categorization level that can be used for the relaxation algorithms"""

    # constraints levels
    VERY_LOW_PRIORITY = 1
    LOW_PRIORITY = 2
    NORMAL_PRIORITY = 3
    MID_PRIORITY = 4
    HIGH_PRIORITY = 5
    VERY_HIGH_PRIORITY = 6
    MANDATORY = 7

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
