from mip.constants import *
from mip.exceptions import SolutionNotAvailable
from mip.expr import LinExpr

class Column:
    """A column contains all the non-zero entries of a variable in the
    constraint matrix. To create a variable see
    :meth:`~mip.model.model.add_var`."""

    def __init__(self,
                 constrs: List["Constr"] = None,
                 coeffs: List[float] = None):
        self.constrs = constrs
        self.coeffs = coeffs


class Var:
    """ Decision variable of the :class:`~mip.model.Model`. The creation of
    variables is performed calling the :meth:`~mip.model.Model.add_var`."""

    def __init__(self,
                 model: Model,
                 idx: int):
        self.__model = model
        self.idx = idx

    def __hash__(self) -> int:
        return self.idx

    def __add__(self, other) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, 1])
        elif isinstance(other, LinExpr):
            return other.__add__(self)
        elif isinstance(other, int) or isinstance(other, float):
            return LinExpr([self], [1], other)

    def __radd__(self, other) -> LinExpr:
        return self.__add__(other)

    def __sub__(self, other) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, -1])
        elif isinstance(other, LinExpr):
            return (-other).__iadd__(self)
        elif isinstance(other, int) or isinstance(other, float):
            return LinExpr([self], [1], -other)

    def __rsub__(self, other) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [-1, 1])
        elif isinstance(other, LinExpr):
            return other.__sub__(self)
        elif isinstance(other, int) or isinstance(other, float):
            return LinExpr([self], [-1], other)

    def __mul__(self, other) -> LinExpr:
        assert isinstance(other, int) or isinstance(other, float)
        return LinExpr([self], [other])

    def __rmul__(self, other) -> LinExpr:
        return self.__mul__(other)

    def __truediv__(self, other) -> LinExpr:
        assert isinstance(other, int) or isinstance(other, float)
        return self.__mul__(1.0 / other)

    def __neg__(self) -> LinExpr:
        return LinExpr([self], [-1.0])

    def __eq__(self, other) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, -1], sense="=")
        elif isinstance(other, LinExpr):
            return other == self
        elif isinstance(other, int) or isinstance(other, float):
            if other != 0:
                return LinExpr([self], [1], -1 * other, sense="=")
            return LinExpr([self], [1], sense="=")

    def __le__(self, other) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, -1], sense="<")
        elif isinstance(other, LinExpr):
            return other >= self
        elif isinstance(other, int) or isinstance(other, float):
            if other != 0:
                return LinExpr([self], [1], -1 * other, sense="<")
            return LinExpr([self], [1], sense="<")

    def __ge__(self, other) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, -1], sense=">")
        elif isinstance(other, LinExpr):
            return other <= self
        elif isinstance(other, int) or isinstance(other, float):
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
    def rc(self) -> float:
        """Reduced cost, only available after a linear programming model (only
        continuous variables) is optimized"""
        if self.__model.status != OptimizationStatus.OPTIMAL:
            raise SolutionNotAvailable('Solution not available.')

        return self.__model.solver.var_get_rc(self)

    @property
    def x(self) -> float:
        """Value of this variable in the solution."""
        if self.__model.status == OptimizationStatus.LOADED:
            raise SolutionNotAvailable('Model was not optimized, \
                solution not available.')
        elif (self.__model.status == OptimizationStatus.INFEASIBLE
              or self.__model.status == OptimizationStatus.CUTOFF):
            raise SolutionNotAvailable('Infeasible __model, \
                solution not available.')
        elif self.__model.status == OptimizationStatus.UNBOUNDED:
            raise SolutionNotAvailable('Unbounded __model, solution not \
                available.')
        elif self.__model.status == OptimizationStatus.NO_SOLUTION_FOUND:
            raise SolutionNotAvailable('Solution not found \
                during optimization.')

        return self.__model.solver.var_get_x(self)

    def xi(self, i: int) -> float:
        """Value for this variable in the :math:`i`-th solution from
        the solution pool."""
        if self.__model.status == OptimizationStatus.LOADED:
            raise SolutionNotAvailable('Model was not optimized, \
                solution not available.')
        elif (self.__model.status == OptimizationStatus.INFEASIBLE or
              self.__model.status == OptimizationStatus.CUTOFF):
            raise SolutionNotAvailable('Infeasible __model, \
                solution not available.')
        elif self.__model.status == OptimizationStatus.UNBOUNDED:
            raise SolutionNotAvailable('Unbounded __model, \
                solution not available.')
        elif self.__model.status == OptimizationStatus.NO_SOLUTION_FOUND:
            raise SolutionNotAvailable('Solution not found \
                during optimization.')

        return self.__model.solver.var_get_xi(self, i)


class VarList(Sequence):
    """ List of model variables (:class:`~mip.model.Var`).

        The number of variables of a model :code:`m` can be queried as
        :code:`len(m.vars)` or as :code:`m.num_cols`.

        Specific variables can be retrieved by their indices or names.
        For example, to print the lower bounds of the first
        variable or of a varible named :code:`z`, you can use, respectively:

        .. code-block:: python

            print(m.vars[0].lb)

        .. code-block:: python

            print(m.vars['z'].lb)
    """

    def __init__(self, model: Model):
        self.__model = model
        self.__vars = []

    def add(self,
            name: str = "",
            lb: float = 0.0,
            ub: float = INF,
            obj: float = 0.0,
            var_type: str = CONTINUOUS,
            column: Column = None) -> Var:
        if not name:
            name = 'var({})'.format(len(self.__vars))
        if var_type == BINARY:
            lb = 0.0
            ub = 1.0
        new_var = Var(self.__model, len(self.__vars))
        self.__model.solver.add_var(obj, lb, ub, var_type, column, name)
        self.__vars.append(new_var)
        return new_var

    def __getitem__(self, key):
        if (isinstance(key, str)):
            return self.__model.var_by_name(key)
        return self.__vars[key]

    def __len__(self) -> int:
        return len(self.__vars)

    def update_vars(self, n_vars: int):
        self.__vars = [Var(self.__model, i) for i in range(n_vars)]

    def remove(self, vars: List[Var]):
        iv = [1 for i in range(len(self.__vars))]
        vlist = [v.idx for v in vars]
        vlist.sort()
        for i in vlist:
            iv[i] = 0
        self.__model.solver.remove_vars(vlist)
        i = 0
        for v in self.__vars:
            if iv[v.idx] == 0:
                v.idx = -1
            else:
                v.idx = i
                i += 1
        self.__vars = [v for v in
                       self.__vars
                       if v.idx != -1]


# same as VarList but does not store
# variables references (used in callbacks)
class VVarList(Sequence):

    def __init__(self, model: Model, start: int = -1, end: int = -1):
        self.__model = model
        if start == -1:
            self.__start = 0
            self.__end = model.solver.num_cols()
        else:
            self.__start = start
            self.__end = end

    def add(self, name: str = "",
            lb: float = 0.0,
            ub: float = INF,
            obj: float = 0.0,
            var_type: str = CONTINUOUS,
            column: Column = None) -> Var:
        solver = self.__model.solver
        if not name:
            name = 'var({})'.format(len(self))
        if var_type == BINARY:
            lb = 0.0
            ub = 1.0
        new_var = Var(self.__model, solver.num_cols())
        solver.add_var(obj, lb, ub, var_type, column, name)
        return new_var

    def __getitem__(self, key):
        if (isinstance(key, str)):
            return self.__model.var_by_name(key)
        if (isinstance(key, slice)):
            return VVarList(self.__model, key.start, key.end)
        if (isinstance(key, int)):
            if key < 0:
                key = self.__end - key
            if key >= self.__end:
                raise IndexError

            return Var(self.__model, key + self.__start)

        raise Exception('Unknow type')

    def __len__(self) -> int:
        return self.__model.solver.num_cols()

