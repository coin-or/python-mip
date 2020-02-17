"""Python-MIP Exceptions"""


class InvalidLinExpr(Exception):
    """Exception that is raised when an invalid
    linear expression is created"""


class InvalidParameter(Exception):
    """Exception that is raised when an invalid/non-existent
    parameter is used or set"""


class ParameterNotAvailable(Exception):
    """Exception that is raised when some parameter is not
    available for the current solver"""


class InfeasibleSolution(Exception):
    """Exception that is raised the produced solution
    is unfeasible"""


class SolutionNotAvailable(Exception):
    """Exception that is raised when a method that requires
    a solution is queried but the solution is not available"""
