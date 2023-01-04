from builtins import property
from typing import List, Optional, Dict, Union, Tuple
import numbers
import mip
from math import fabs
import math


class Column:
    """A column contains all the non-zero entries of a variable in the
    constraint matrix. To create a variable see
    :meth:`~mip.Model.add_var`."""

    __slots__ = ["constrs", "coeffs"]

    def __init__(
        self,
        constrs=None,  # type : Optional[List["mip.Constr"]]
        coeffs=None,  # type: Optional[List[numbers.Real]]
    ):
        self.constrs = constrs
        self.coeffs = coeffs

    def __str__(self) -> str:
        res = "["
        for k in range(len(self.constrs)):
            res += "{} {}".format(self.coeffs[k], self.constrs[k].name)
            if k < len(self.constrs) - 1:
                res += ", "
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

    If used in intermediate calculations, the solved value of the linear
    expression can be obtained with the ``x`` parameter, just as with
    a ``Var``.

    .. code:: python

     a = 10*x1 + 7*x4
     print(a.x)

    """

    __slots__ = ["__const", "__expr", "__sense"]

    def __init__(
        self,
        variables: Optional[List["mip.Var"]] = None,
        coeffs: Optional[List[numbers.Real]] = None,
        const: numbers.Real = 0.0,
        sense: str = "",
        expr: Optional[Dict["mip.Var", numbers.Real]] = None,
    ):
        self.__const = const
        self.__expr = {}  # type: Dict[mip.Var, numbers.Real]
        self.__sense = sense

        if variables is not None and coeffs is not None:
            if len(variables) != len(coeffs):
                raise ValueError("Coefficients and variables must be same length.")
            if expr is not None:
                raise ValueError(
                    "You should pass eiter 'expr' or 'variables and coeffs' to the"
                    "constructor, not the three simultaneously."
                )
            self.__expr = dict(zip(variables, coeffs))

        elif expr is not None:
            self.__expr = expr.copy()

    def __add__(
        self,
        other: Union["mip.Var", "mip.LinExpr", numbers.Real],
    ) -> "mip.LinExpr":
        if isinstance(other, numbers.Real) and fabs(other) < mip.EPS:
            return self

        result = self.copy()
        if isinstance(other, Var):
            result.add_var(other, 1)
        elif isinstance(other, LinExpr):
            result.add_expr(other)
        elif isinstance(other, numbers.Real):
            result.add_const(other)
        else:
            raise TypeError("type {} not supported".format(type(other)))
        return result

    def __radd__(
        self,
        other: Union["mip.Var", "mip.LinExpr", numbers.Real],
    ) -> "mip.LinExpr":
        return self.__add__(other)

    def __sub__(
        self,
        other: Union["mip.Var", "mip.LinExpr", numbers.Real],
    ) -> "mip.LinExpr":
        result = self.copy()
        if isinstance(other, Var):
            result.add_var(other, -1)
        elif isinstance(other, LinExpr):
            result.add_expr(other, -1)
        elif isinstance(other, numbers.Real):
            result.add_const(-other)
        else:
            raise TypeError("type {} not supported".format(type(other)))
        return result

    def __rsub__(
        self,
        other: Union["mip.Var", "mip.LinExpr", numbers.Real],
    ) -> "mip.LinExpr":
        return (-self).__add__(other)

    def __mul__(self, other: numbers.Real) -> "mip.LinExpr":
        if not isinstance(other, numbers.Real):
            raise TypeError("Can not multiply with type {}".format(type(other)))

        if fabs(other - 1) < mip.EPS:
            return self

        result = self.copy()
        result.__const *= other
        for var in result.__expr.keys():
            result.__expr[var] *= other
        return result

    def __rmul__(self, other: numbers.Real) -> "mip.LinExpr":
        return self.__mul__(other)

    def __truediv__(self, other: numbers.Real) -> "mip.LinExpr":
        if not isinstance(other, numbers.Real):
            raise TypeError("Can not divide with type {}".format(type(other)))
        if fabs(other) < mip.EPS:
            raise ZeroDivisionError("Expression division by zero")
        return self.__mul__(1.0 / other)

    def __neg__(self) -> "LinExpr":
        return self.__mul__(-1)

    def __str__(self) -> str:
        result = []

        if hasattr(self, "__sense"):
            if self.__sense == mip.MINIMIZE:
                result.append("Minimize ")
            elif self.__sense == mip.MAXIMIZE:
                result.append("Minimize ")

        if self.__expr:
            for var, coeff in self.__expr.items():
                result.append("+ " if coeff >= 0 else "- ")
                result.append(str(abs(coeff)) if abs(coeff) != 1 else "")
                result.append("{var} ".format(**locals()))

        if self.__sense:
            if self.__sense == mip.EQUAL:
                result.append(" = ")
            if self.__sense == mip.LESS_OR_EQUAL:
                result.append(" <= ")
            if self.__sense == mip.GREATER_OR_EQUAL:
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

    def __eq__(self, other) -> "LinExpr":
        result = self - other
        result.__sense = "="
        return result

    def __le__(
        self,
        other: Union["mip.Var", "LinExpr", numbers.Real],
    ) -> "mip.LinExpr":
        result = self - other
        result.__sense = "<"
        return result

    def __ge__(
        self,
        other: Union["mip.Var", "LinExpr", numbers.Real],
    ) -> "mip.LinExpr":
        result = self - other
        result.__sense = ">"
        return result

    def __len__(self):
        return len(self.__expr)

    def add_const(self, val: numbers.Real):
        """adds a constant value to the linear expression, in the case of
        a constraint this corresponds to the right-hand-side

        Args:
            val(numbers.Real): a real number
        """
        self.__const += val

    def add_expr(self, expr: "LinExpr", coeff: numbers.Real = 1):
        """Extends a linear expression with the contents of another.

        Args:
            expr (LinExpr): another linear expression
            coeff (numbers.Real): coefficient which will multiply the linear
                expression added
        """
        self.__const += expr.const * coeff
        for var, coeff_var in expr.expr.items():
            self.add_var(var, coeff_var * coeff)

    def add_term(
        self,
        term: Union["mip.Var", "mip.LinExpr", numbers.Real],
        coeff: numbers.Real = 1,
    ):
        """Adds a term to the linear expression.

        Args:
            term (Union[mip.Var, LinExpr, numbers.Real]) : can be a
                variable, another linear expression or a real number.

            coeff (numbers.Real) : coefficient which will multiply the added
                term

        """
        if isinstance(term, Var):
            self.add_var(term, coeff)
        elif isinstance(term, LinExpr):
            self.add_expr(term, coeff)
        elif isinstance(term, numbers.Real):
            self.add_const(term * coeff)
        else:
            raise TypeError("type {} not supported".format(type(term)))

    def add_var(self, var: "mip.Var", coeff: numbers.Real = 1):
        """Adds a variable with a coefficient to the linear expression.

        Args:
            var (mip.Var) : a variable
            coeff (numbers.Real) : coefficient which the variable will be added
        """
        self.__expr.setdefault(var, 0)
        self.__expr[var] += coeff

    def set_expr(self: "LinExpr", expr: Dict["mip.Var", numbers.Real]):
        """Sets terms of the linear expression

        Args:
            expr(Dict[mip.Var, numbers.Real]) : dictionary mapping variables to
                their coefficients in the linear expression.
        """

        self.__expr = expr

    def copy(self) -> "mip.LinExpr":
        return LinExpr(const=self.__const, sense=self.__sense, expr=self.__expr)

    def equals(self, other: "mip.LinExpr") -> bool:
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
    def const(self) -> numbers.Real:
        """constant part of the linear expression"""
        return self.__const

    @property
    def expr(self) -> Dict["mip.Var", numbers.Real]:
        """the non-constant part of the linear expression

        Dictionary with pairs: (variable, coefficient) where coefficient
        is a real number.

        :rtype: Dict[mip.Var, numbers.Real]
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
    def violation(self) -> Optional[numbers.Real]:
        """Amount that current solution violates this constraint

        If a solution is available, than this property indicates how much
        the current solution violates this constraint.
        """
        # No violation can be computed for something that isn't a constraint
        # or has no solution yet
        if self.sense == "" or self.x is None:
            return None

        lhs = sum(coef * var.x for (var, coef) in self.__expr.items())
        rhs = -self.const

        if self.sense == "=":
            viol = abs(lhs - rhs)
        elif self.sense == "<":
            viol = max(lhs - rhs, 0.0)
        elif self.sense == ">":
            viol = max(rhs - lhs, 0.0)
        else:
            raise ValueError("Invalid sense {}".format(self.sense))

        return viol

    @property
    def x(self) -> Optional[numbers.Real]:
        """Value of this linear expression in the solution. None
        is returned if no solution is available."""
        x = self.__const
        for var, coef in self.__expr.items():
            var_x = var.x
            if var_x is None:
                return None
            x += var_x * coef
        return x

    def __float__(self):
        x = self.x
        return math.nan if x is None else float(x)

    @property
    def model(self) -> Optional["mip.Model"]:
        """Model which this LinExpr refers to, None if no variables are
        involved.

        :rtype: Optional[mip.Model]
        """
        if not self.expr:
            return None

        return next(iter(self.expr)).model


class Constr:
    """A row (constraint) in the constraint matrix.

    A constraint is a specific :class:`~LinExpr` that includes a
    sense (<, > or == or less-or-equal, greater-or-equal and equal,
    respectively) and a right-hand-side constant value. Constraints can be
    added to the model using the overloaded operator :code:`+=` or using
    the method :meth:`~mip.Model.add_constr` of the
    :class:`~mip.Model` class:

    .. code:: python

      m += 3*x1 + 4*x2 <= 5

    summation expressions are also supported:

    .. code:: python

      m += xsum(x[i] for i in range(n)) == 1
    """

    __slots__ = ["__model", "idx", "__priority"]

    def __init__(
        self,
        model: "mip.Model",
        idx: int,
        priority: "mip.constants.ConstraintPriority" = None,
    ):
        self.__model = model
        self.idx = idx
        self.__priority = priority

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
        else:
            raise ValueError("Invalid sense {}".format(self.expr.sense))

        return res

    @property
    def rhs(self) -> numbers.Real:
        """The right-hand-side (constant value) of the linear constraint."""
        return self.__model.solver.constr_get_rhs(self.idx)

    @rhs.setter
    def rhs(self, rhs: numbers.Real):
        self.__model.solver.constr_set_rhs(self.idx, rhs)

    @property
    def slack(self) -> Optional[numbers.Real]:
        """Value of the slack in this constraint in the optimal
        solution. Available only if the formulation was solved.
        """
        return self.__model.solver.constr_get_slack(self)

    @property
    def pi(self) -> Optional[numbers.Real]:
        """Value for the dual variable of this constraint in the optimal
        solution of a linear programming :class:`~mip.Model`. Only
        available if a pure linear programming problem was solved (only
        continuous variables).
        """
        return self.__model.solver.constr_get_pi(self)

    @property
    def expr(self) -> LinExpr:
        """Linear expression that defines the constraint.

        :rtype: mip.LinExpr"""
        return self.__model.solver.constr_get_expr(self)

    @expr.setter
    def expr(self, value: LinExpr):
        self.__model.solver.constr_set_expr(self, value)

    @property
    def name(self) -> str:
        """constraint name"""
        return self.__model.solver.constr_get_name(self.idx)

    @property
    def priority(self) -> "mip.constants.ConstraintPriority":
        """priority value"""
        return self.__priority

    @priority.setter
    def priority(self, priority: "mip.constants.ConstraintPriority"):
        self.__priority = priority


class Var:
    """Decision variable of the :class:`~mip.Model`. The creation of
    variables is performed calling the :meth:`~mip.Model.add_var`."""

    __slots__ = ["_model", "_idx"]

    def __init__(self, model: "mip.Model", idx: int):
        self._model = model
        self._idx = idx

    def __hash__(self) -> int:
        return self._idx

    def __add__(
        self, other: Union["mip.Var", LinExpr, numbers.Real]
    ) -> Union["mip.Var", LinExpr]:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, 1])
        if isinstance(other, LinExpr):
            return other.__add__(self)
        if isinstance(other, numbers.Real):
            if fabs(other) < mip.EPS:
                return self
            return LinExpr([self], [1], other)

        raise TypeError("type {} not supported".format(type(other)))

    def __radd__(
        self, other: Union["mip.Var", LinExpr, numbers.Real]
    ) -> Union["mip.Var", LinExpr]:
        return self.__add__(other)

    def __sub__(
        self, other: Union["mip.Var", LinExpr, numbers.Real]
    ) -> Union["mip.Var", LinExpr]:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, -1])
        if isinstance(other, LinExpr):
            return (-other).__add__(self)
        if isinstance(other, numbers.Real):
            if fabs(other) < mip.EPS:
                return self
            return LinExpr([self], [1], -other)

        raise TypeError("type {} not supported".format(type(other)))

    def __rsub__(
        self, other: Union["mip.Var", LinExpr, numbers.Real]
    ) -> Union["mip.Var", LinExpr]:
        if isinstance(other, Var):
            return LinExpr([self, other], [-1, 1])
        if isinstance(other, LinExpr):
            return other.__sub__(self)
        if isinstance(other, numbers.Real):
            return LinExpr([self], [-1], other)

        raise TypeError("type {} not supported".format(type(other)))

    def __mul__(self, other: numbers.Real) -> LinExpr:
        if not isinstance(other, numbers.Real):
            raise TypeError("Can not multiply with type {}".format(type(other)))
        return LinExpr([self], [other])

    def __rmul__(self, other: numbers.Real) -> LinExpr:
        return self.__mul__(other)

    def __truediv__(self, other: numbers.Real) -> LinExpr:
        if not isinstance(other, numbers.Real):
            raise TypeError("Can not divide with type {}".format(type(other)))
        if abs(other) < mip.EPS:
            raise ZeroDivisionError("Variable division by zero")
        return self.__mul__(1.0 / other)

    def __neg__(self) -> LinExpr:
        return LinExpr([self], [-1.0])

    def __eq__(self, other) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, -1], sense="=")
        if isinstance(other, LinExpr):
            return LinExpr([self], [1]) == other
        if isinstance(other, numbers.Real):
            return LinExpr([self], [1], -1 * other, sense="=")

        raise TypeError("type {} not supported".format(type(other)))

    def __le__(self, other: Union["mip.Var", LinExpr, numbers.Real]) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, -1], sense="<")
        if isinstance(other, LinExpr):
            return LinExpr([self], [1]) <= other
        if isinstance(other, numbers.Real):
            return LinExpr([self], [1], -1 * other, sense="<")

        raise TypeError("type {} not supported".format(type(other)))

    def __ge__(self, other: Union["mip.Var", LinExpr, numbers.Real]) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, -1], sense=">")
        if isinstance(other, LinExpr):
            return LinExpr([self], [1]) >= other
        if isinstance(other, numbers.Real):
            return LinExpr([self], [1], -1 * other, sense=">")

        raise TypeError("type {} not supported".format(type(other)))

    @property
    def name(self) -> str:
        """Variable name."""
        return self._model.solver.var_get_name(self.idx)

    def __str__(self) -> str:
        return self.name

    @property
    def lb(self) -> numbers.Real:
        """Variable lower bound."""
        return self._model.solver.var_get_lb(self)

    @lb.setter
    def lb(self, value: numbers.Real):
        self._model.solver.var_set_lb(self, value)

    @property
    def ub(self) -> numbers.Real:
        """Variable upper bound."""
        return self._model.solver.var_get_ub(self)

    @ub.setter
    def ub(self, value: numbers.Real):
        self._model.solver.var_set_ub(self, value)

    @property
    def obj(self) -> numbers.Real:
        """Coefficient of variable in the objective function."""
        return self._model.solver.var_get_obj(self)

    @obj.setter
    def obj(self, value: numbers.Real):
        self._model.solver.var_set_obj(self, value)

    @property
    def branch_priority(self) -> numbers.Real:
        """
        Variable's branching priority in the branch and bound process.
        Note: variables with higher priority are selected first. Default value is zero.
        """
        return self._model.solver.var_get_branch_priority(self)

    @branch_priority.setter
    def branch_priority(self, value: numbers.Real):
        self._model.solver.var_set_branch_priority(self, value)

    @property
    def var_type(self) -> str:
        """Variable type, ('B') BINARY, ('C') CONTINUOUS and ('I') INTEGER."""
        return self._model.solver.var_get_var_type(self)

    @var_type.setter
    def var_type(self, value: str):
        if value not in (mip.BINARY, mip.CONTINUOUS, mip.INTEGER):
            raise ValueError(
                "Expected one of {}, but got {}".format(
                    (mip.BINARY, mip.CONTINUOUS, mip.INTEGER), value
                )
            )
        self._model.solver.var_set_var_type(self, value)

    @property
    def column(self) -> Column:
        """Variable coefficients in constraints.

        :rtype: mip.Column
        """
        return self._model.solver.var_get_column(self)

    @column.setter
    def column(self, value: Column):
        self._model.solver.var_set_column(self, value)

    @property
    def rc(self) -> Optional[numbers.Real]:
        """Reduced cost, only available after a linear programming model (only
        continuous variables) is optimized. Note that None is returned if no
        optimum solution is available"""

        return self._model.solver.var_get_rc(self)

    @property
    def x(self) -> Optional[numbers.Real]:
        """Value of this variable in the solution. Note that None is returned
        if no solution is not available."""
        return self._model.solver.var_get_x(self)

    def xi(self, i: int) -> Optional[numbers.Real]:
        """Value for this variable in the :math:`i`-th solution from the solution
        pool. Note that None is returned if the solution is not available."""
        if self._model.status in [
            mip.OptimizationStatus.OPTIMAL,
            mip.OptimizationStatus.FEASIBLE,
        ]:
            return self._model.solver.var_get_xi(self, i)
        return None

    def __float__(self):
        x = self.x
        return math.nan if x is None else float(x)

    @property
    def model(self) -> "mip.Model":
        """Model which this variable refers to.

        :rtype: mip.Model
        """
        return self._model

    @property
    def idx(self) -> int:
        """Internal index of the variable to the model.

        :rtype: int
        """
        return self._idx


class ConflictGraph:

    r"""A conflict graph stores conflicts between incompatible assignments in
    binary variables.

    For example, if there is a constraint :math:`x_1 + x_2 \leq 1` then
    there is a conflict between :math:`x_1 = 1` and :math:`x_2 = 1`. We can state
    that :math:`x_1` and :math:`x_2` are conflicting. Conflicts can also involve the complement
    of a binary variable. For example, if there is a constraint :math:`x_1 \leq
    x_2` then there is a conflict between :math:`x_1 = 1` and :math:`x_2 = 0`.
    We now can state that :math:`x_1` and :math:`\lnot x_2` are conflicting."""

    __slots__ = ["model"]

    def __init__(self, model: "mip.Model"):
        self.model = model
        self.model.solver.update_conflict_graph()

    @property
    def density(self) -> float:
        return self.model.solver.cgraph_density()

    def conflicting(
        self,
        e1: Union["mip.LinExpr", "mip.Var"],
        e2: Union["mip.LinExpr", "mip.Var"],
    ) -> bool:
        """Checks if two assignments of binary variables are in conflict.

        Args:
            e1 (Union[mip.LinExpr, mip.Var]): binary variable, if assignment to be
                tested is the assignment to one, or a linear expression like x == 0
                to indicate that conflict with the complement of the variable
                should be tested.

            e2 (Union[mip.LinExpr, mip.Var]): binary variable, if assignment to be
                tested is the assignment to one, or a linear expression like x == 0
                to indicate that conflict with the complement of the variable
                should be tested.
        """
        if not isinstance(e1, (mip.LinExpr, mip.Var)):
            raise TypeError("type {} not supported".format(type(e1)))
        if not isinstance(e2, (mip.LinExpr, mip.Var)):
            raise TypeError("type {} not supported".format(type(e2)))

        return e1.model.solver.conflicting(e1, e2)

    def conflicting_assignments(
        self, v: Union["mip.LinExpr", "mip.Var"]
    ) -> Tuple[List["mip.Var"], List["mip.Var"]]:
        """Returns from the conflict graph all assignments conflicting with one
        specific assignment.

        Args:
            v (Union[mip.Var, mip.LinExpr]): binary variable, if assignment to be
                tested is the assignment to one or a linear expression like x == 0
                to indicate the complement.

        :rtype: Tuple[List[mip.Var], List[mip.Var]]

        Returns:
            Returns a tuple with two lists. The first one indicates variables
            whose conflict occurs when setting them to one. The second list
            includes variable whose conflict occurs when setting them to zero.
        """
        if not isinstance(v, (mip.LinExpr, mip.Var)):
            raise TypeError("type {} not supported".format(type(v)))

        return self.model.solver.conflicting_nodes(v)
