class InvalidLinExpr(Exception):
    """Exception that is raised when an invalid
    linear expression is created"""

    pass


class InvalidParameter(Exception):
    """Exception that is raised when an invalid/non-existent
    parameter is used or set"""

    pass


class ParameterNotAvailable(Exception):
    """Exception that is raised when some parameter is not
    available for the current solver"""

    pass


class InfeasibleSolution(Exception):
    """Exception that is raised the produced solution
    is unfeasible"""

    pass
