class InvalidLinExpr(Exception):
    """Exception that is raised when an invalid
    linear expression is created"""
    pass


class SolutionNotAvailable(Exception):
    """Exception that is raised when some method to query some
    solution property is used but no solution is available"""
    pass
