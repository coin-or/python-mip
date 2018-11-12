from typing import Dict, List

from milp.constants import *


class Column:

    def __init__(self, constrs: List["Constr"] = None, coeffs: List[float] = None):
        self.constrs: List[Constr] = constrs if constrs else []
        self.coeffs: List[float] = coeffs if coeffs else []


class Constr:

    def __init__(self, model: "Model", idx: int, name: str = ""):
        self.model: Model = model
        self.idx: int = idx
        self.name: str = name  # discuss this var

    def __hash__(self) -> int:
        return self.idx

    def __str__(self) -> str:
        return self.name

    @property
    def pi(self) -> float:
        return self.model.solver.constr_get_pi(self)

    @property
    def row(self) -> "Row":
        return self.model.solver.constr_get_row(self)

    @row.setter
    def row(self, value: "Row") -> None:
        self.model.solver.constr_set_row(self, value)


class LinExpr:

    def __init__(self,
                 variables: List["Var"] = None,
                 coeffs: List[float] = None,
                 const: float = 0,
                 sense: str = ""):
        self.const: int = const
        self.expr: Dict[Var, float] = {}
        self.sense: str = sense

        if variables:
            assert len(variables) == len(coeffs)
            for i in range(len(coeffs)):
                if coeffs[i] == 0:
                    continue
                self.add_var(variables[i], coeffs[i])

    def __add__(self, other) -> "LinExpr":
        result: LinExpr = self.copy()
        if isinstance(other, Var):
            result.add_var(other, 1)
        elif isinstance(other, LinExpr):
            result.add_expr(other)
        elif isinstance(other, (int, float)):
            result.add_const(other)
        return result

    def __radd__(self, other) -> "LinExpr":
        return self.__add__(other)

    def __iadd__(self, other) -> "LinExpr":
        if isinstance(other, Var):
            self.add_var(other, 1)
        elif isinstance(other, LinExpr):
            self.add_expr(other)
        elif isinstance(other, (int, float)):
            self.add_const(other)
        return self

    def __sub__(self, other) -> "LinExpr":
        result: LinExpr = self.copy()
        if isinstance(other, Var):
            result.add_var(other, -1)
        elif isinstance(other, LinExpr):
            result.add_expr(-other)
        elif isinstance(other, (int, float)):
            result.add_const(-other)
        return result

    def __rsub__(self, other) -> "LinExpr":
        return self.__add__(-other)

    def __isub__(self, other) -> "LinExpr":
        if isinstance(other, Var):
            self.add_var(other, -1)
        elif isinstance(other, LinExpr):
            self.add_expr(-other)
        elif isinstance(other, (int, float)):
            self.add_const(-other)
        return self

    def __mul__(self, other) -> "LinExpr":
        assert isinstance(other, int) or isinstance(other, float)
        result: LinExpr = self.copy()
        for var in result.expr.keys():
            result.expr[var] *= other
        return result

    def __rmul__(self, other) -> "LinExpr":
        return self.__mul__(other)

    def __imul__(self, other) -> "LinExpr":
        assert isinstance(other, int) or isinstance(other, float)
        for var in self.expr.keys():
            self.expr[var] *= other
        return self

    def __truediv__(self, other) -> "LinExpr":
        assert isinstance(other, int) or isinstance(other, float)
        result: LinExpr = self.copy()
        for var in result.expr.keys():
            result.expr[var] /= other
        return result

    def __itruediv__(self, other) -> "LinExpr":
        assert isinstance(other, int) or isinstance(other, float)
        for var in self.expr.keys():
            self.expr[var] /= other
        return self

    def __neg__(self) -> "LinExpr":
        return self.__mul__(-1)

    def __str__(self) -> str:
        result: List[str] = []

        if self.expr:
            for var, coeff in self.expr.items():
                result.append("+ " if coeff >= 0 else "- ")
                result.append(str(abs(coeff)) if abs(coeff) != 1 else "")
                result.append("{var} ".format(**locals()))

        if self.sense:
            result.append(self.sense + "= ")
            result.append(str(abs(self.const)) if self.const < 0 else "+ " + str(abs(self.const)))
        elif self.const != 0:
            result.append(
                "+ " + str(abs(self.const)) if self.const > 0 else "- " + str(abs(self.const)))

        return "".join(result)

    def __eq__(self, other) -> "LinExpr":
        result: LinExpr = self - other
        result.sense = "="
        return result

    def __le__(self, other) -> "LinExpr":
        result: LinExpr = self - other
        result.sense = "<"
        return result

    def __ge__(self, other) -> "LinExpr":
        result: LinExpr = self - other
        result.sense = ">"
        return result

    def add_const(self, const: float):
        self.const += const

    def add_expr(self, expr: "LinExpr", coeff: float = 1):
        for var, coeff_var in expr.expr.items():
            self.add_var(var, coeff_var * coeff)

    def add_term(self, expr, coeff: float = 1):
        if isinstance(expr, Var):
            self.add_var(expr, coeff)
        elif isinstance(expr, LinExpr):
            self.add_expr(expr, coeff)
        elif isinstance(expr, float) or isinstance(expr, int):
            self.add_const(expr)

    def add_var(self, var: "Var", coeff: float = 1):
        if var in self.expr:
            if -EPS <= self.expr[var] + coeff <= EPS:
                del self.expr[var]
            else:
                self.expr[var] += coeff
        else:
            self.expr[var] = coeff

    def copy(self) -> "LinExpr":
        copy: LinExpr = LinExpr()
        copy.const = self.const
        copy.expr = self.expr.copy()
        copy.sense = self.sense
        return copy


class Model:

    def __init__(self, name: str = "",
                 sense: str = MINIMIZE,
                 solver_name: str = GUROBI):
        # initializing variables with default values
        self.name: str = name
        self.sense: str = sense
        self.solver_name: str = solver_name
        self.solver: Solver = None

        # list of constraints and variables
        self.constrs: List[Constr] = []
        self.vars: List[Var] = []

        # todo: implement code to detect solver automatically
        if solver_name == GUROBI:
            from milp.gurobi import SolverGurobi
            self.solver = SolverGurobi(name, sense)

    def __del__(self):
        if self.solver:
            del self.solver

    def __iadd__(self, other) -> 'Model':
        if isinstance(other, LinExpr):
            if len(other.sense) == 0:
                # adding objective function components
                self.set_objective(other)
            else:
                # adding constraint
                self.add_constr(other)
        elif isinstance(other, tuple):
            if isinstance(other[0], LinExpr) and isinstance(other[1], str):
                if len(other[0].sense) == 0:
                    self.set_objective(other[0])
                else:
                    self.add_constr(other[0], other[1])

        return self

    def add_var(self, name: str = "",
                lb: float = 0.0,
                ub: float = INF,
                obj: float = 0.0,
                type: str = CONTINUOUS,
                column: "Column" = None) -> "Var":
        idx = self.solver.add_var(obj, lb, ub, type, column, name)
        self.vars.append(Var(self, idx, name))
        return self.vars[-1]

    def add_constr(self, lin_expr: "LinExpr", name: str = "") -> Constr:
        if isinstance(lin_expr, bool):
            return None  # empty constraint
        idx = self.solver.add_constr(lin_expr, name)
        self.constrs.append(Constr(self, idx, name))
        return self.constrs[-1]

    def copy(self, solver_name: str = None) -> "Model":
        if not solver_name:
            solver_name = self.solver_name
        copy: Model = Model(self.name, self.sense, solver_name)

        # adding variables
        for v in self.vars:
            copy.add_var(name=v.name, lb=v.lb, ub=v.ub, obj=0.0, type=v.type)

        # adding constraints
        for c in self.constrs:
            expr = LinExpr()  # todo: make copy of constraint's lin_expr
            copy.add_constr(lin_expr=expr, name=c.name)

        # setting objective function's constant
        expr = LinExpr()  # todo: make copy of objective's lin_expr
        copy.set_objective(expr)

        return copy

    def optimize(self):
        self.solver.optimize()

    def set_start(self, variables: List["Var"], values: List[float]):
        self.solver.set_start(variables, values)

    def set_objective(self, expr, sense: str = ""):
        if isinstance(expr, int) or isinstance(expr, float):
            self.solver.set_objective(LinExpr([], [], expr))
        elif isinstance(expr, Var):
            self.solver.set_objective(LinExpr([expr], [1]))
        elif isinstance(expr, LinExpr):
            self.solver.set_objective(expr, sense)

    def write(self, path: str):
        self.solver.write(path)


class Row:

    def __init__(self, vars: List["Var"] = None, coeffs: List[float] = None):
        self.vars: List[Var] = vars if vars else []
        self.coeffs: List[Constr] = coeffs if coeffs else []


class Solver:

    def __init__(self, name: str, sense: str):
        self.name: str = name
        self.sense: str = sense

    def __del__(self): pass

    def add_var(self,
                name: str = "",
                obj: float = 0,
                lb: float = 0,
                ub: float = INF,
                type: str = CONTINUOUS,
                column: "Column" = None) -> int: pass

    def add_constr(self, lin_expr: "LinExpr", name: str = "") -> int: pass

    def optimize(self) -> int: pass

    def set_start(self, variables: List["Var"], values: List[float]) -> None: pass

    def set_objective(self, lin_expr: "LinExpr", sense: str = "") -> None: pass

    def write(self, file_path: str) -> None: pass

    # Constraint-related getters/setters

    def constr_get_pi(self, constr: Constr) -> float: pass

    def constr_get_row(self, constr: Constr) -> Row: pass

    def constr_set_row(self, constr: Constr, value: Row) -> Row: pass

    # Variable-related getters/setters

    def var_get_lb(self, var: "Var") -> float: pass

    def var_set_lb(self, var: "Var", value: float) -> None: pass

    def var_get_ub(self, var: "Var") -> float: pass

    def var_set_ub(self, var: "Var", value: float) -> None: pass

    def var_get_obj(self, var: "Var") -> float: pass

    def var_set_obj(self, var: "Var", value: float) -> None: pass

    def var_get_type(self, var: "Var") -> str: pass

    def var_set_type(self, var: "Var", value: str) -> None: pass

    def var_get_column(self, var: "Var") -> Column: pass

    def var_set_column(self, var: "Var", value: Column) -> None: pass

    def var_get_rc(self, var: "Var") -> float: pass

    def var_get_x(self, var: "Var") -> float: pass


class Var:

    def __init__(self,
                 model: Model,
                 idx: int,
                 name: str = ""):
        self.model: Model = model
        self.idx: int = idx
        self.name: str = name  # discuss this var

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
            return other.__rsub__(self)
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
        if other != 0:
            return LinExpr([self], [1], -1 * other, sense="=")
        return LinExpr([self], [1], sense="=")

    def __le__(self, other) -> LinExpr:
        if other != 0:
            return LinExpr([self], [1], -1 * other, sense="<")
        return LinExpr([self], [1], sense="<")

    def __ge__(self, other) -> LinExpr:
        if other != 0:
            return LinExpr([self], [1], -1 * other, sense=">")
        return LinExpr([self], [1], sense=">")

    def __str__(self) -> str:
        return self.name

    @property
    def lb(self) -> float:
        return self.model.solver.var_get_lb(self)

    @lb.setter
    def lb(self, value: float):
        self.model.solver.var_set_lb(self, value)

    @property
    def ub(self) -> float:
        return self.model.solver.var_get_ub(self)

    @ub.setter
    def ub(self, value: float):
        self.model.solver.var_set_ub(self, value)

    @property
    def obj(self) -> float:
        return self.model.solver.var_get_obj(self)

    @obj.setter
    def obj(self, value: float):
        self.model.solver.var_set_obj(self, value)

    @property
    def type(self) -> str:
        return self.model.solver.var_get_type(self)

    @type.setter
    def type(self, value: str):
        assert value in (BINARY, CONTINUOUS, INTEGER)
        self.model.solver.var_set_type(self, value)

    @property
    def column(self) -> Column:
        return self.model.solver.var_get_column(self)

    @column.setter
    def column(self, value: Column):
        self.model.solver.var_set_column(self, value)

    @property
    def rc(self) -> float:
        return self.model.solver.var_get_rc(self)

    @property
    def x(self) -> float:
        return self.model.solver.var_get_x(self)


def xsum(terms) -> LinExpr:
    result: LinExpr = LinExpr()
    for term in terms:
        result.add_term(term)
    return result


# function aliases
quicksum = xsum

# vim: ts=4 sw=4 et
