"""Python-MIP Exceptions"""


class MipBaseException(Exception):
    """Base class for all exceptions specific to Python MIP. Only sub-classes
    of this exception are raised.
    Inherits from the Python builtin ``Exception``."""


class ProgrammingError(MipBaseException):
    """Exception that is raised when the calling program performs an invalid
     or nonsensical operation.
     Inherits from :attr:`~mip.exceptions.MipBaseException`."""


class InterfacingError(MipBaseException):
    """Exception that is raised when an unknown error occurs while interfacing
    with a solver.
    Inherits from :attr:`~mip.exceptions.MipBaseException`."""


class InvalidLinExpr(MipBaseException):
    """Exception that is raised when an invalid
    linear expression is created.
    Inherits from :attr:`~mip.exceptions.MipBaseException`."""


class InvalidParameter(MipBaseException):
    """Exception that is raised when an invalid/non-existent
    parameter is used or set.
    Inherits from :attr:`~mip.exceptions.MipBaseException`."""


class ParameterNotAvailable(MipBaseException):
    """Exception that is raised when some parameter is not
    available or can not be set.
    Inherits from :attr:`~mip.exceptions.MipBaseException`."""


class InfeasibleSolution(MipBaseException):
    """Exception that is raised the produced solution
    is unfeasible.
    Inherits from :attr:`~mip.exceptions.MipBaseException`."""


class SolutionNotAvailable(MipBaseException):
    """Exception that is raised when a method that requires
    a solution is queried but the solution is not available.
    Inherits from :attr:`~mip.exceptions.MipBaseException`."""
