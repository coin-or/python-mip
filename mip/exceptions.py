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


class SolutionNotAvailable(Exception):
    """Exception that is raised when some method to query some
    solution property is used but no solution is available"""
    pass
