from builtins import property
from typing import List, Optional, Dict, Union, TYPE_CHECKING

from mip.constants import (
    BINARY,
    CONTINUOUS,
    INTEGER,
    OptimizationStatus,
    EPS,
    MAXIMIZE,
    MINIMIZE,
    EQUAL,
    LESS_OR_EQUAL,
    GREATER_OR_EQUAL,
)

if TYPE_CHECKING:
    from mip.model import Model


class Column:
    """A column contains all the non-zero entries of a variable in the
    constraint matrix. To create a variable see
    :meth:`~mip.model.Model.add_var`."""

    def __init__(self, constrs: List["Constr"] = [], coeffs: List[float] = []):
        self.constrs = constrs
        self.coeffs = coeffs

    def __str__(self) -> str:
        res = "["
        for k in range(len(self.constrs)):
            res += "{}: {}, ".format(self.constrs[k].idx, self.coeffs[k])
        res += "]"
        return res


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

    def __init__(
        self,
        variables: List["Var"] = [],
        coeffs: List[float] = [],
        const: float = 0.0,
        sense: str = "",
    ):
        self.__const = const
        self.__expr = {}  # type: Dict[Var, float]
        self.__sense = sense

        if variables:
            assert len(variables) == len(coeffs)
            for i in range(len(coeffs)):
                if abs(coeffs[i]) <= 1e-12:
                    continue
                self.add_var(variables[i], coeffs[i])

    def __add__(
        self: "LinExpr", other: Union["Var", "LinExpr", int, float]
    ) -> "LinExpr":
        result = self.copy()
        if isinstance(other, Var):
            result.add_var(other, 1)
        elif isinstance(other, LinExpr):
            result.add_expr(other)
        elif isinstance(other, (int, float)):
            result.add_const(other)
        return result

    def __radd__(
        self: "LinExpr", other: Union["Var", "LinExpr", int, float]
    ) -> "LinExpr":
        return self.__add__(other)

    def __iadd__(
        self: "LinExpr", other: Union["Var", "LinExpr", int, float]
    ) -> "LinExpr":
        if isinstance(other, Var):
            self.add_var(other, 1)
        elif isinstance(other, LinExpr):
            self.add_expr(other)
        elif isinstance(other, (int, float)):
            self.add_const(other)
        return self

    def __sub__(
        self: "LinExpr", other: Union["Var", "LinExpr", int, float]
    ) -> "LinExpr":
        result = self.copy()
        if isinstance(other, Var):
            result.add_var(other, -1)
        elif isinstance(other, LinExpr):
            result.add_expr(other, -1)
        elif isinstance(other, (int, float)):
            result.add_const(-other)
        return result

    def __rsub__(
        self: "LinExpr", other: Union["Var", "LinExpr", int, float]
    ) -> "LinExpr":
        return (-self).__add__(other)

    def __isub__(
        self: "LinExpr", other: Union["Var", "LinExpr", int, float]
    ) -> "LinExpr":
        if isinstance(other, Var):
            self.add_var(other, -1)
        elif isinstance(other, LinExpr):
            self.add_expr(other, -1)
        elif isinstance(other, (int, float)):
            self.add_const(-other)
        return self

    def __mul__(self: "LinExpr", other: Union[int, float]) -> "LinExpr":
        assert isinstance(other, (int, float))
        result = self.copy()
        result.__const *= other
        for var in result.__expr.keys():
            result.__expr[var] *= other
        return result

    def __rmul__(self: "LinExpr", other: Union[int, float]) -> "LinExpr":
        return self.__mul__(other)

    def __imul__(self: "LinExpr", other: Union[int, float]) -> "LinExpr":
        assert isinstance(other, (int, float))
        self.__const *= other
        for var in self.__expr.keys():
            self.__expr[var] *= other
        return self

    def __truediv__(self: "LinExpr", other: Union[int, float]) -> "LinExpr":
        assert isinstance(other, (int, float))
        result = self.copy()
        result.__const /= other
        for var in result.__expr.keys():
            result.__expr[var] /= other
        return result

    def __itruediv__(self: "LinExpr", other: Union[int, float]) -> "LinExpr":
        assert isinstance(other, (int, float))
        self.__const /= other
        for var in self.__expr.keys():
            self.__expr[var] /= other
        return self

    def __neg__(self: "LinExpr") -> "LinExpr":
        return self.__mul__(-1)

    def __str__(self: "LinExpr") -> str:
        result = []

        if hasattr(self, "__sense"):
            if self.__sense == MINIMIZE:
                result.append("Minimize ")
            elif self.__sense == MAXIMIZE:
                result.append("Minimize ")

        if self.__expr:
            for var, coeff in self.__expr.items():
                result.append("+ " if coeff >= 0 else "- ")
                result.append(str(abs(coeff)) if abs(coeff) != 1 else "")
                result.append("{var} ".format(**locals()))

        if hasattr(self, "__sense"):
            if self.__sense == EQUAL:
                result.append(" = ")
            if self.__sense == LESS_OR_EQUAL:
                result.append(" <= ")
            if self.__sense == GREATER_OR_EQUAL:
                result.append(" >= ")
            result.append(
                str(abs(self.__const))
                if self.__const < 0
                else "- " + str(abs(self.__const))
            )
        elif self.__const != 0:
            result.append(
                "+ " + str(abs(self.__const))
                if self.__const > 0
                else "- " + str(abs(self.__const))
            )

        return "".join(result)

    def __eq__(self: "LinExpr", other) -> "LinExpr":
        result = self - other
        result.__sense = "="
        return result

    def __le__(
        self: "LinExpr", other: Union["Var", "LinExpr", int, float]
    ) -> "LinExpr":
        result = self - other
        result.__sense = "<"
        return result

    def __ge__(
        self: "LinExpr", other: Union["Var", "LinExpr", int, float]
    ) -> "LinExpr":
        result = self - other
        result.__sense = ">"
        return result

    def add_const(self: "LinExpr", __const: float):
        """adds a constant value to the linear expression, in the case of
        a constraint this correspond to the right-hand-side"""
        self.__const += __const

    def add_expr(self: "LinExpr", __expr: "LinExpr", coeff: float = 1):
        """extends a linear expression with the contents of another"""
        self.__const += __expr.__const * coeff
        for var, coeff_var in __expr.__expr.items():
            self.add_var(var, coeff_var * coeff)

    def add_term(
        self: "LinExpr",
        __expr: Union["Var", "LinExpr", int, float],
        coeff: float = 1,
    ):
        """extends a linear expression with another multiplied by a constant
        value coefficient"""
        if isinstance(__expr, Var):
            self.add_var(__expr, coeff)
        elif isinstance(__expr, LinExpr):
            self.add_expr(__expr, coeff)
        elif isinstance(__expr, (int, float)):
            self.add_const(__expr)

    def add_var(self: "LinExpr", var: "Var", coeff: float = 1):
        """adds a variable with a coefficient to the constraint"""
        if var in self.__expr:
            if -EPS <= self.__expr[var] + coeff <= EPS:
                del self.__expr[var]
            else:
                self.__expr[var] += coeff
        else:
            self.__expr[var] = coeff

    def copy(self: "LinExpr") -> "LinExpr":
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

    def __hash__(self: "LinExpr"):
        hash_el = [v.idx for v in self.__expr.keys()]
        for c in self.__expr.values():
            hash_el.append(c)
        hash_el.append(self.__const)
        hash_el.append(self.__sense)
        return hash(tuple(hash_el))

    @property
    def const(self: "LinExpr") -> float:
        """constant part of the linear expression"""
        return self.__const

    @property
    def expr(self: "LinExpr") -> Dict["Var", float]:
        """the non-constant part of the linear expression

        Dictionary with pairs: (variable, coefficient) where coefficient
        is a float.
        """
        return self.__expr

    @property
    def sense(self: "LinExpr") -> str:
        """sense of the linear expression

        sense can be EQUAL("="), LESS_OR_EQUAL("<"), GREATER_OR_EQUAL(">") or
        empty ("") if this is an affine expression, such as the objective
        function
        """
        return self.__sense

    @sense.setter
    def sense(self: "LinExpr", value):
        """sense of the linear expression

        sense can be EQUAL("="), LESS_OR_EQUAL("<"), GREATER_OR_EQUAL(">") or
        empty ("") if this is an affine expression, such as the objective
        function
        """
        self.__sense = value

    @property
    def violation(self: "LinExpr"):
        """Amount that current solution violates this constraint

        If a solution is available, than this property indicates how much
        the current solution violates this constraint.
        """
        lhs = sum(coef * var.x for (var, coef) in self.__expr.items())
        rhs = -self.const
        viol = 0.0
        if self.sense == "=":
            viol = abs(lhs - rhs)
        elif self.sense == "<":
            viol = max(lhs - rhs, 0.0)
        elif self.sense == ">":
            viol = max(rhs - lhs, 0.0)

        return viol


class Constr:
    """ A row (constraint) in the constraint matrix.

        A constraint is a specific :class:`~mip.entities.LinExpr` that includes a
        sense (<, > or == or less-or-equal, greater-or-equal and equal,
        respectively) and a right-hand-side constant value. Constraints can be
        added to the model using the overloaded operator :code:`+=` or using
        the method :meth:`~mip.model.Model.add_constr` of the
        :class:`~mip.model.Model` class:

        .. code:: python

          m += 3*x1 + 4*x2 <= 5

        summation expressions are also supported:

        .. code:: python

          m += xsum(x[i] for i in range(n)) == 1
    """

    def __init__(self, model: "Model", idx: int):
        self.__model = model
        self.idx = idx

    def __hash__(self) -> int:
        return self.idx

    def __str__(self) -> str:
        if self.name:
            res = self.name + ":"
        else:
            res = "constr({}): ".format(self.idx + 1)
        line = ""
        len_line = 0
        for (var, val) in self.expr.expr.items():
            astr = " {:+} {}".format(val, var.name)
            len_line += len(astr)
            line += astr

            if len_line > 75:
                line += "\n\t"
                len_line = 0
        res += line
        rhs = self.expr.const * -1.0
        if self.expr.sense == "=":
            res += " = {}".format(rhs)
        elif self.expr.sense == "<":
            res += " <= {}".format(rhs)
        elif self.expr.sense == ">":
            res += " >= {}".format(rhs)

        return res

    @property
    def rhs(self) -> float:
        """The right-hand-side (constant value) of the linear constraint."""
        return self.__model.solver.constr_get_rhs(self.idx)

    @rhs.setter
    def rhs(self, rhs: float):
        self.__model.solver.constr_set_rhs(self.idx, rhs)

    @property
    def slack(self) -> Optional[float]:
        """Value of the slack in this constraint in the optimal
        solution. Available only if the formulation was solved.
        """
        return self.__model.solver.constr_get_slack(self)

    @property
    def pi(self) -> Optional[float]:
        """Value for the dual variable of this constraint in the optimal
        solution of a linear programming :class:`~mip.model.Model`. Only
        available if a pure linear programming problem was solved (only
        continuous variables).
        """
        if (
            self.__model.status != OptimizationStatus.OPTIMAL
            or self.__model.num_int > 0
        ):
            return None
        return self.__model.solver.constr_get_pi(self)

    @property
    def expr(self) -> LinExpr:
        """contents of the constraint"""
        return self.__model.solver.constr_get_expr(self)

    @expr.setter
    def expr(self, value: LinExpr):
        self.__model.solver.constr_set_expr(self, value)

    @property
    def name(self) -> str:
        """constraint name"""
        return self.__model.solver.constr_get_name(self.idx)


class Var:
    """ Decision variable of the :class:`~mip.model.Model`. The creation of
    variables is performed calling the :meth:`~mip.model.Model.add_var`."""

    def __init__(self, model: "Model", idx: int):
        self.__model = model
        self.idx = idx

    def __hash__(self) -> int:
        return self.idx

    def __add__(self, other: Union["Var", LinExpr, int, float]) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, 1])
        elif isinstance(other, LinExpr):
            return other.__add__(self)
        elif isinstance(other, (int, float)):
            return LinExpr([self], [1], other)

    def __radd__(self, other: Union["Var", LinExpr, int, float]) -> LinExpr:
        return self.__add__(other)

    def __sub__(self, other: Union["Var", LinExpr, int, float]) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, -1])
        elif isinstance(other, LinExpr):
            return (-other).__iadd__(self)
        elif isinstance(other, (int, float)):
            return LinExpr([self], [1], -other)

    def __rsub__(self, other: Union["Var", LinExpr, int, float]) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [-1, 1])
        elif isinstance(other, LinExpr):
            return other.__sub__(self)
        elif isinstance(other, (int, float)):
            return LinExpr([self], [-1], other)

    def __mul__(self, other: Union[int, float]) -> LinExpr:
        assert isinstance(other, (int, float))
        return LinExpr([self], [other])

    def __rmul__(self, other: Union[int, float]) -> LinExpr:
        return self.__mul__(other)

    def __truediv__(self, other: Union[int, float]) -> LinExpr:
        assert isinstance(other, (int, float))
        return self.__mul__(1.0 / other)

    def __neg__(self) -> LinExpr:
        return LinExpr([self], [-1.0])

    def __eq__(self, other) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, -1], sense="=")
        elif isinstance(other, LinExpr):
            return other == self
        elif isinstance(other, (int, float)):
            if other != 0:
                return LinExpr([self], [1], -1 * other, sense="=")
            return LinExpr([self], [1], sense="=")

    def __le__(self, other: Union["Var", LinExpr, int, float]) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, -1], sense="<")
        elif isinstance(other, LinExpr):
            return other >= self
        elif isinstance(other, (int, float)):
            if other != 0:
                return LinExpr([self], [1], -1 * other, sense="<")
            return LinExpr([self], [1], sense="<")

    def __ge__(self, other: Union["Var", LinExpr, int, float]) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, -1], sense=">")
        elif isinstance(other, LinExpr):
            return other <= self
        elif isinstance(other, (int, float)):
            if other != 0:
                return LinExpr([self], [1], -1 * other, sense=">")
            return LinExpr([self], [1], sense=">")

    @property
    def name(self) -> str:
        """Variable name."""
        return self.__model.solver.var_get_name(self.idx)

    def __str__(self) -> str:
        return self.name

    @property
    def lb(self) -> float:
        """Variable lower bound."""
        return self.__model.solver.var_get_lb(self)

    @lb.setter
    def lb(self, value: float):
        self.__model.solver.var_set_lb(self, value)

    @property
    def ub(self) -> float:
        """Variable upper bound."""
        return self.__model.solver.var_get_ub(self)

    @ub.setter
    def ub(self, value: float):
        self.__model.solver.var_set_ub(self, value)

    @property
    def obj(self) -> float:
        """Coefficient of variable in the objective function."""
        return self.__model.solver.var_get_obj(self)

    @obj.setter
    def obj(self, value: float):
        self.__model.solver.var_set_obj(self, value)

    @property
    def var_type(self) -> str:
        """Variable type: ('B') BINARY, ('C') CONTINUOUS and ('I') INTEGER."""
        return self.__model.solver.var_get_var_type(self)

    @var_type.setter
    def var_type(self, value: str):
        assert value in (BINARY, CONTINUOUS, INTEGER)
        self.__model.solver.var_set_var_type(self, value)

    @property
    def column(self) -> Column:
        """Variable coefficients in constraints."""
        return self.__model.solver.var_get_column(self)

    @column.setter
    def column(self, value: Column):
        self.__model.solver.var_set_column(self, value)

    @property
    def rc(self) -> Optional[float]:
        """Reduced cost, only available after a linear programming model (only
        continuous variables) is optimized. Note that None is returned if no
        optimum solution is available"""
        if (
            self.__model.status != OptimizationStatus.OPTIMAL
            or self.__model.num_int > 0
        ):
            return None

        return self.__model.solver.var_get_rc(self)

    @property
    def x(self) -> Optional[float]:
        """Value of this variable in the solution. Note that None is returned
        if no solution is not available."""
        if self.__model.status in [
            OptimizationStatus.OPTIMAL,
            OptimizationStatus.FEASIBLE,
        ]:
            return self.__model.solver.var_get_x(self)
        return None

    def xi(self, i: int) -> Optional[float]:
        """Value for this variable in the :math:`i`-th solution from the solution
        pool. Note that None is returned if the solution is not available."""
        if self.__model.status in [
            OptimizationStatus.OPTIMAL,
            OptimizationStatus.FEASIBLE,
        ]:
            return self.__model.solver.var_get_xi(self, i)
        return None
