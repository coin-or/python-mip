"""Different entities in the model (variables, constraints)"""
from builtins import property
from typing import List, Optional, Union

from mip.constants import BINARY, CONTINUOUS, INTEGER, \
        OptimizationStatus
from mip.exceptions import SolutionNotAvailable


class Column:
    """A column contains all the non-zero entries of a variable in the
    constraint matrix. To create a variable see
    :meth:`~mip.model.Model.add_var`."""

    def __init__(self,
                 constrs: Optional[List["Constr"]] = None,
                 coeffs: Optional[List[float]] = None):
        self.constrs = constrs
        self.coeffs = coeffs


class LinExpr:
    """
    Linear expressions are used to express the model constraints and optionally
    the objective function. These expressions are created using operators and
    variables. Summation of variables with coefficients can be included using the
    function xsum.

    Consider a :class:`~mip.model.Model` object m, the objective function of :code:`m` can be
    specified as:

    .. code:: python

     m.objective = 10*x1 + 7*x4

    In the example bellow, a constraint is added to the model.

    .. code:: python

     m += xsum(3*x[i] i in range(n)) - xsum(x[i] i in range(m))

    A constraint is just a linear expression with the addition of a sense (==,
    <= or >=) and a right hand side, e.g.:

    .. code:: python

     m += x1 + x2 + x3 == 1
    """

    def __init__(self,
                 variables: Optional[List[Union["Var", int]]] = None,
                 coeffs: Optional[List[float]] = None,
                 const: float = 0.0,
                 sense: str = "",
                 model: Optional["Model"] = None):
        assert isinstance(const, (int, float))
        self.__const = const
        self.__model = model
        if variables:
            if isinstance(variables[0], Var):
                self.__idx: List[int] = [var.idx for var in variables]
                self.__model = variables[0].model
            elif isinstance(variables[0], int):
                self.__idx: List[int] = variables.copy()
                assert self.__model is not None
            self.__coef: List[float] = [coef for coef in coeffs]
        else:
            self.__idx: List[int] = []
            self.__coef: List[float] = []
        self.__sense = sense

    @property
    def idx(self) -> List[int]:
        """List of variable indexes in the linear expression."""
        return self.__idx

    @property
    def coef(self) -> List[float]:
        """List of variable coefficients in the liner expression."""
        return self.__coef

    @property
    def model(self) -> Optional["Model"]:
        """Model that this linear expression refers to."""
        return self.__model

    def __add__(self, other: Union["Var", "LinExpr", int, float]) -> "LinExpr":
        result = self.copy()
        if isinstance(other, Var):
            result.add_var(other, 1)
        elif isinstance(other, LinExpr):
            result.add_expr(other)
        elif isinstance(other, (int, float)):
            result.add_const(other)
        return result

    def __radd__(self, other: Union["Var", "LinExpr", int, float]) -> "LinExpr":
        return self.__add__(other)

    def __iadd__(self, other: Union["Var", "LinExpr", int, float]) -> "LinExpr":
        if isinstance(other, Var):
            self.add_var(other, 1)
        elif isinstance(other, LinExpr):
            self.add_expr(other)
        elif isinstance(other, (int, float)):
            self.add_const(other)
        return self

    def __sub__(self, other: Union["Var", "LinExpr", int, float]) -> "LinExpr":
        result = self.copy()
        if isinstance(other, Var):
            result.add_var(other, -1)
            return result
        if isinstance(other, LinExpr):
            result.add_expr(other, -1)
            return result

        assert isinstance(other, (int, float))
        result.add_const(-other)
        return result

    def __rsub__(self, other: Union["Var", "LinExpr", int, float]) -> "LinExpr":
        return (-self).__add__(other)

    def __isub__(self, other: Union["Var", "LinExpr", int, float]) -> "LinExpr":
        if isinstance(other, Var):
            self.add_var(other, -1)
        elif isinstance(other, LinExpr):
            self.add_expr(other, -1)
        elif isinstance(other, (int, float)):
            self.add_const(-other)
        return self

    def __mul__(self, other: Union[int, float]) -> "LinExpr":
        assert isinstance(other, (float, int))
        result = self.copy()
        result.__const *= other
        for i in range(len(result.__coef)):
            result.__coef[i] *= other

        return result

    def __rmul__(self, other: Union[int, float]) -> "LinExpr":
        return self.__mul__(other)

    def __imul__(self, other: Union[int, float]) -> "LinExpr":
        assert isinstance(other, (int, float))
        self.__const *= other
        for i in range(len(self.__coef)):
            self.__coef[i] *= other
        return self

    def __truediv__(self, other: Union[int, float]) -> "LinExpr":
        assert isinstance(other, (int, float))
        result = self.copy()
        result.__const /= other
        for i in range(len(result.__coef)):
            result.__coef[i] /= other
        return result

    def __itruediv__(self, other: Union[int, float]) -> "LinExpr":
        assert isinstance(other, (int, float))
        self.__const /= other
        for i in range(len(self.__coef)):
            self.__coef[i] /= other
        return self

    def __neg__(self) -> "LinExpr":
        return self.__mul__(-1)

    def __str__(self) -> str:
        result = ""

        for i, idx in enumerate(self.__idx):
            coeff = self.__coef[i]
            result += "+ " if coeff >= 0 else "- "
            result += str(abs(coeff)) if abs(coeff) != 1 else ""
            result += ' %s ' % self.__model.vars[idx].name

        if self.__sense:
            result += self.__sense + "= "
            result += (str(abs(self.__const)) if self.__const < 0 else
                       "- " + str(abs(self.__const)))
        elif self.__const != 0:
            result += ("+ " + str(abs(self.__const)) if self.__const > 0
                       else "- " + str(abs(self.__const)))

        return result

    def __len__(self) -> int:
        return len(self.idx)

    def __eq__(self, other: Union["Var", "LinExpr", int, float]) -> "LinExpr":
        result = self - other
        result.sense = "="
        return result

    def __le__(self, other: Union["Var", "LinExpr", int, float]) -> "LinExpr":
        result = self - other
        result.sense = "<"
        return result

    def __ge__(self, other: Union["Var", "LinExpr", int, float]) -> "LinExpr":
        result = self - other
        result.sense = ">"
        return result

    def add_const(self, __const: Union[int, float]):
        """adds a constant value to the linear expression, in the case of
        a constraint this correspond to the right-hand-side"""
        assert isinstance(__const, (int, float))
        self.__const += __const

    def add_expr(self, __expr: "LinExpr", coeff: Union[int, float] = 1):
        """extends a linear expression with the contents of another"""
        self.__const += __expr.const * coeff
        for i, idx in enumerate(__expr.idx):
            self.__idx.append(idx)
            self.__coef.append(__expr.coef[i]*coeff)
        if __expr.model:
            self.__model = __expr.model

    def add_term(self, expr: Union["Var", "LinExpr", int, float],
                 coeff: Union[int, float] = 1):
        """extends a linear expression with another multiplied by a constant
        value coefficient"""
        if isinstance(expr, Var):
            self.add_var(expr, coeff)
            return
        if isinstance(expr, LinExpr):
            self.add_expr(expr, coeff)
            return

        # int, float
        assert isinstance(expr, (int, float))
        self.add_const(expr)

    def add_var(self, var: "Var", coeff: Union[int, float] = 1.0):
        """adds a variable with a coefficient to the constraint"""
        self.__idx.append(var.idx)
        self.__coef.append(float(coeff))
        self.__model = var.model

    def pack(self, iv: Optional[List[int]]):
        """Groups variable's coefficients. This is automatically called before
            adding a linear expression to the model."""
        if iv is None:
            iv = [-1 for i in range(len(self.__model.num_cols))]
        else:
            if self.__model.num_cols > len(iv):
                iv = [-1 for i in range(max(len(self.__model.num_cols),
                                            len(iv)*2))]

        new_idx: List[int] = []
        new_coef: List[float] = []
        for i, idx in enumerate(self.__idx):
            if iv[idx] == -1:
                iv[idx] = len(new_idx)
                new_idx.append(idx)
                new_coef.append(self.__coef[i])
            else:
                new_coef[iv[idx]] += self.__coef[i]

        # clearing incidence vector, since it may be reused
        for idx in new_idx:
            iv[idx] = -1

        # after grouping variables coefficients, some coefficients
        # may be zero, removing those at the end
        while new_coef and abs(new_coef[-1]) <= 1e-20:
            new_coef.pop()
            new_idx.pop()

        self.__idx = new_idx
        self.__coef = new_coef

        # removing elements with zero coefficient in the middle of the vector,
        # if still any

        i = 0
        while i < len(self.__coef):
            if abs(self.__coef[i]) <= 1e-20:
                if i < len(self.__coef)-1:
                    self.__idx[i], self.__idx[-1] = \
                        self.__idx[-1], self.__idx[i]
                    self.__coef[i], self.__coef[-1] = \
                        self.__coef[-1], self.__coef[i]
                self.__idx.pop()
                self.__coef.pop()
            else:
                i += 1

    def copy(self) -> "LinExpr":
        """Returns a copy of the LinExpr object."""
        return LinExpr(self.__idx, self.__coef, self.__const, self.__sense,
                       self.__model)

    def equals(self: "LinExpr", other: "LinExpr") -> bool:
        """returns true if a linear expression equals to another,
        false otherwise"""
        if self.__sense != other.sense:
            return False
        if len(self.__idx) != len(other.idx):
            return False
        for (i, idx) in enumerate(self.__idx):
            if idx != other.idx[i]:
                return False
        for (i, coef) in enumerate(self.__coef):
            if abs(coef - other.coef[i]) > 1e-12:
                return False
        return True

    def __hash__(self):
        hash_el = [self.__const, self.__sense]
        hash_el += self.__idx
        hash_el += self.__coef
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
        return {self.__model.vars[self.__idx[i]]: self.__coef[i] for i in
                range(len(self.__idx))}

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
        self.__sense = value

    @property
    def violation(self):
        """Amount that current solution violates this constraint

        If a solution is available, than this property indicates how much
        the current solution violates this constraint.
        """
        lhs = sum(self.__coef[i] * self.__model.vars[idx].x
                  for i, idx in enumerate(self.__idx))
        rhs = -self.__const
        viol = 0.0
        if self.sense == '=':
            viol = abs(lhs - rhs)
        elif self.sense == '<':
            viol = max(lhs - rhs, 0.0)
        elif self.sense == '>':
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
            res = self.name + ':'
        else:
            res = 'constr({}): '.format(self.idx + 1)
        line = ''
        len_line = 0
        for (var, val) in self.expr.expr.items():
            astr = ' {:+} {}'.format(val, var.name)
            len_line += len(astr)
            line += astr

            if len_line > 75:
                line += '\n\t'
                len_line = 0
        res += line
        rhs = self.expr.const * -1.0
        if self.expr.sense == '=':
            res += ' = {}'.format(rhs)
        elif self.expr.sense == '<':
            res += ' <= {}'.format(rhs)
        elif self.expr.sense == '>':
            res += ' >= {}'.format(rhs)

        return res

    @property
    def slack(self) -> float:

        """Value of the slack in this constraint in the optimal
        solution. Available only if the formulation was solved.
        """

        return self.__model.solver.constr_get_slack(self)

    @property
    def pi(self) -> float:

        """Value for the dual variable of this constraint in the optimal
        solution of a linear programming :class:`~mip.model.Model`. Only
        available if a pure linear programming problem was solved (only
        continuous variables).
        """

        return self.__model.solver.constr_get_pi(self)

    @property
    def expr(self) -> "LinExpr":
        """contents of the constraint"""
        return self.__model.solver.constr_get_expr(self)

    @expr.setter
    def expr(self, value: "LinExpr"):
        self.__model.solver.constr_set_expr(self, value)

    @property
    def name(self) -> str:
        """constraint name"""
        return self.__model.solver.constr_get_name(self.idx)



class Var:
    """ Decision variable of the :class:`~mip.model.Model`. The creation of
    variables is performed calling the :meth:`~mip.model.Model.add_var`."""

    def __init__(self,
                 model: "Model",
                 idx: int):
        self.__model = model
        self.idx = idx

    @property
    def model(self) -> "Model":
        """Model that this variable refers to"""
        return self.__model

    def __hash__(self) -> int:
        return self.idx

    def __add__(self, other: Union["Var", "LinExpr", float, int]) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self.idx, other.idx], [1.0, 1.0], 0.0,
                           "", self.model)
        if isinstance(other, LinExpr):
            return other.__add__(self)

        # int or float
        assert isinstance(other, (int, float))
        return LinExpr([self.idx], [1.0], other, "", self.model)

    def __radd__(self, other) -> LinExpr:
        return self.__add__(other)

    def __sub__(self, other: Union["Var", "LinExpr", float, int]) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self.idx, other.idx], [1.0, -1.0], 0.0,
                           "", self.model)
        if isinstance(other, LinExpr):
            return (-other).__iadd__(self)

        # int or float
        return LinExpr([self.idx], [1.0], -other, "", self.model)

    def __rsub__(self, other: Union["Var", "LinExpr", float, int]) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self.idx, other.idx], [-1.0, 1.0], 0.0,
                           "", self.model)
        if isinstance(other, LinExpr):
            return other.__sub__(self)

        # int or float
        assert isinstance(other, (int, float))
        return LinExpr([self.idx], [-1.0], other, "", self.model)

    def __mul__(self, other: Union[int, float]) -> LinExpr:
        return LinExpr([self.idx], [other], 0.0, "", self.model)

    def __rmul__(self, other: Union[int, float]) -> LinExpr:
        return self.__mul__(other)

    def __truediv__(self, other: Union[int, float]) -> LinExpr:
        return self.__mul__(1.0 / other)

    def __neg__(self) -> LinExpr:
        return LinExpr([self.idx], [-1.0], 0.0, "", self.model)

    def __eq__(self, other: Union["Var", "LinExpr", float, int]) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self.idx, other.idx], [1, -1], 0.0, "=", self.model)
        if isinstance(other, LinExpr):
            return other == self

        # int or float
        assert isinstance(other, (int, float))
        if other != 0:
            return LinExpr([self.idx], [1], -1 * other, "=", self.model)
        return LinExpr([self], [1.0], 0.0, "=", self.model)

    def __le__(self, other: Union["Var", "LinExpr", float, int]) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self.idx, other.idx], [1, -1], 0.0, "<", self.model)
        if isinstance(other, LinExpr):
            return other >= self

        # int or float
        assert isinstance(other, (int, float))
        if other != 0:
            return LinExpr([self.idx], [1], -1 * other, "<", self.model)
        return LinExpr([self.idx], [1], 0.0, "<", self.model)

    def __ge__(self, other: Union["Var", "LinExpr", float, int]) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self.idx, other.idx], [1, -1], 0.0, ">", self.model)
        if isinstance(other, LinExpr):
            return other <= self

        # int or float
        assert isinstance(other, (int, float))
        if other != 0:
            return LinExpr([self], [1], -1 * other, ">", self.model)
        return LinExpr([self], [1], 0.0, ">", self.model)

    @property
    def name(self) -> str:
        """Variable name."""
        return self.__model.solver.var_get_name(self.idx)

    def __str__(self) -> str:
        return 'var: (name=%s, lb=%g, ub=%g, obj=%g)' % (self.name, self.lb, self.ub, self.obj)

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
    def rc(self) -> float:
        """Reduced cost, only available after a linear programming model (only
        continuous variables) is optimized"""
        if self.__model.status != OptimizationStatus.OPTIMAL:
            raise SolutionNotAvailable('Solution not available.')

        return self.__model.solver.var_get_rc(self)

    @property
    def x(self) -> Optional[float]:
        """Value of this variable in the solution."""
        if self.__model.status in [OptimizationStatus.OPTIMAL,
                                   OptimizationStatus.FEASIBLE]:
            return self.__model.solver.var_get_x(self)

        return None

    def xi(self, i: int) -> Optional[float]:
        """Value for this variable in the :math:`i`-th solution from
        the solution pool."""
        if self.__model.status in [OptimizationStatus.OPTIMAL,
                                   OptimizationStatus.FEASIBLE]:
            return self.__model.solver.var_get_xi(self, i)

        return None
