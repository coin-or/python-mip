from mip.constants import *
from typing import List, Tuple
import mip.var as var


class LinExpr:
    """
    Linear expressions are used to enter the objective function and the model \
    constraints. These expressions are created using operators and variables.

    Consider a model object m, the objective function of :code:`m` can be
    specified as:

    .. code:: python

     m.objective = 10*x1 + 7*x4

    In the example bellow, a constraint is added to the model

    .. code:: python

     m += xsum(3*x[i] i in range(n)) - xsum(x[i] i in range(m))

    A constraint is just a linear expression with the addition of a sense (==,
    <= or >=) and a right hand side, e.g.:

    .. code:: python

     m += x1 + x2 + x3 == 1
    """

    def __init__(self,
                 variables: List[var.Var] = None,
                 coeffs: List[float] = None,
                 const: float = 0.0,
                 sense: str = ""):
        self.__const = const
        self.__expr = {}
        self.__sense = sense

        if variables:
            assert len(variables) == len(coeffs)
            for i in range(len(coeffs)):
                if abs(coeffs[i]) <= 1e-12:
                    continue
                self.add_var(variables[i], coeffs[i])

    def __add__(self, other) -> "LinExpr":
        result = self.copy()
        if isinstance(other, var.Var):
            result.add_var(other, 1)
        elif isinstance(other, LinExpr):
            result.add_expr(other)
        elif isinstance(other, (int, float)):
            result.add_const(other)
        return result

    def __radd__(self, other) -> "LinExpr":
        return self.__add__(other)

    def __iadd__(self, other) -> "LinExpr":
        if isinstance(other, var.Var):
            self.add_var(other, 1)
        elif isinstance(other, LinExpr):
            self.add_expr(other)
        elif isinstance(other, (int, float)):
            self.add_const(other)
        return self

    def __sub__(self, other) -> "LinExpr":
        result = self.copy()
        if isinstance(other, var.Var):
            result.add_var(other, -1)
        elif isinstance(other, LinExpr):
            result.add_expr(other, -1)
        elif isinstance(other, (int, float)):
            result.add_const(-other)
        return result

    def __rsub__(self, other) -> "LinExpr":
        return (-self).__add__(other)

    def __isub__(self, other) -> "LinExpr":
        if isinstance(other, var.Var):
            self.add_var(other, -1)
        elif isinstance(other, LinExpr):
            self.add_expr(other, -1)
        elif isinstance(other, (int, float)):
            self.add_const(-other)
        return self

    def __mul__(self, other) -> "LinExpr":
        assert isinstance(other, (float, int))
        result = self.copy()
        result.__const *= other
        for var in result.__expr.keys():
            result.__expr[var] *= other

        # if constraint __sense will change
        if self.__sense == GREATER_OR_EQUAL and other <= -1e-8:
            self.__sense = LESS_OR_EQUAL
        if self.__sense == LESS_OR_EQUAL and other <= -1e-8:
            self.__sense = GREATER_OR_EQUAL

        return result

    def __rmul__(self, other) -> "LinExpr":
        return self.__mul__(other)

    def __imul__(self, other) -> "LinExpr":
        assert isinstance(other, (int, float))
        self.__const *= other
        for var in self.__expr.keys():
            self.__expr[var] *= other
        return self

    def __truediv__(self, other) -> "LinExpr":
        assert isinstance(other, (int, float))
        result = self.copy()
        result.__const /= other
        for var in result.__expr.keys():
            result.__expr[var] /= other
        return result

    def __itruediv__(self, other) -> "LinExpr":
        assert isinstance(other, int) or isinstance(other, float)
        self.__const /= other
        for var in self.__expr.keys():
            self.__expr[var] /= other
        return self

    def __neg__(self) -> "LinExpr":
        return self.__mul__(-1)

    def __str__(self) -> str:
        result = []

        if self.__expr:
            for var, coeff in self.__expr.items():
                result.append("+ " if coeff >= 0 else "- ")
                result.append(str(abs(coeff)) if abs(coeff) != 1 else "")
                result.append("{var} ".format(**locals()))

        if self.__sense:
            result.append(self.__sense + "= ")
            result.append(str(abs(self.__const)) if self.__const < 0 else
                          "- " + str(abs(self.__const)))
        elif self.__const != 0:
            result.append(
                "+ " + str(abs(self.__const)) if self.__const > 0
                else "- " + str(abs(self.__const)))

        return "".join(result)

    def __eq__(self, other) -> "LinExpr":
        result = self - other
        result.__sense = "="
        return result

    def __le__(self, other) -> "LinExpr":
        result = self - other
        result.__sense = "<"
        return result

    def __ge__(self, other) -> "LinExpr":
        result = self - other
        result.__sense = ">"
        return result

    def add_const(self, __const: float):
        """adds a constant value to the linear expression, in the case of
        a constraint this correspond to the right-hand-side"""
        self.__const += __const

    def add_expr(self, __expr: "LinExpr", coeff: float = 1):
        """extends a linear expression with the contents of another"""
        self.__const += __expr.__const * coeff
        for var, coeff_var in __expr.__expr.items():
            self.add_var(var, coeff_var * coeff)

    def add_term(self, __expr, coeff: float = 1):
        """extends a linear expression with another multiplied by a constant
        value coefficient"""
        if isinstance(__expr, var.Var):
            self.add_var(__expr, coeff)
        elif isinstance(__expr, LinExpr):
            self.add_expr(__expr, coeff)
        elif isinstance(__expr, float) or isinstance(__expr, int):
            self.add_const(__expr)

    def add_var(self, var: var.Var, coeff: float = 1):
        """adds a variable with a coefficient to the constraint"""
        if var in self.__expr:
            if -EPS <= self.__expr[var] + coeff <= EPS:
                del self.__expr[var]
            else:
                self.__expr[var] += coeff
        else:
            self.__expr[var] = coeff

    def copy(self) -> "LinExpr":
        copy = LinExpr()
        copy.__const = self.__const
        copy.__expr = self.__expr.copy()
        copy.__sense = self.__sense
        return copy

    def equals(self: "LinExpr", other: "LinExpr") -> bool:
        """returns true if a linear expression equals to another,
        false otherwise"""
        if self.__sense != other.__sense:
            return False
        if len(self.__expr) != len(other.__expr):
            return False
        if abs(self.__const - other.__const) >= 1e-12:
            return False
        other_contents = {vr.idx: coef for vr, coef in other.__expr.items()}
        for (v, c) in self.__expr.items():
            if v.idx not in other_contents:
                return False
            oc = other_contents[v.idx]
            if abs(c - oc) > 1e-12:
                return False
        return True

    def __hash__(self):
        hash_el = [v.idx for v in self.__expr.keys()]
        for c in self.__expr.values():
            hash_el.append(c)
        hash_el.append(self.__const)
        hash_el.append(self.__sense)
        return hash(tuple(hash_el))

    @property
    def const(self) -> float:
        """constant part of the linear expression"""
        return self.__const

    @property
    def expr(self) -> dict:
        """the non-constant part of the linear expression

        Dictionary with pairs: (variable, coefficient) where coefficient
        is a float.
        """
        return self.__expr

    @property
    def sense(self) -> str:
        """sense of the linear expression

        sense can be EQUAL("="), LESS_OR_EQUAL("<"), GREATER_OR_EQUAL(">") or
        empty ("") if this is an affine expression, such as the objective
        function
        """
        return self.__sense

    @sense.setter
    def sense(self, value):
        """sense of the linear expression

        sense can be EQUAL("="), LESS_OR_EQUAL("<"), GREATER_OR_EQUAL(">") or
        empty ("") if this is an affine expression, such as the objective
        function
        """
        self.__sense = value

    @property
    def violation(self):
        """Amount that current solution violates this constraint

        If a solution is available, than this property indicates how much
        the current solution violates this constraint.
        """
        lhs = sum(coef * var.x for (var, coef) in self.__expr.items())
        rhs = -self.const
        viol = 0.0
        if self.sense == '=':
            viol = abs(lhs - rhs)
        elif self.sense == '<':
            viol = max(lhs - rhs, 0.0)
        elif self.sense == '>':
            viol = max(rhs - lhs, 0.0)

        return viol
