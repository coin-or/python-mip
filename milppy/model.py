from typing import Dict, List

# epsilon number (practical zero)
EPS = 10e-6

# infinity representation
INF = float("inf")

# optimization status
ERROR = -1
OPTIMAL = 0
INFEASIBLE = 1
UNBOUNDED = 2
FEASIBLE = 3
INT_INFEASIBLE = 4
NO_SOLUTION_FOUND = 5
LOADED = 6
CUTOFF = 7

# constraint senses
EQUAL = "="
LESS_OR_EQUAL = "<"
GREATER_OR_EQUAL = ">"

# optimization directions
MINIMIZE = "MIN"
MAXIMIZE = "MAX"

# variable types
BINARY = "B"
CONTINUOUS = "C"
INTEGER = "I"

# solvers
CBC = "CBC"
CPLEX = "CPX"
GUROBI = "GRB"
SCIP = "SCIP"


class Column:

    def __init__(self, constrs: List["Constr"] = None,
                 coeffs: List[float] = None):
        self.constrs = constrs if constrs else []
        self.coeffs = coeffs if coeffs else []


class Constr:

    def __init__(self, model: "Model",
                 idx: int,
                 name: str = ""):
        self.model = model
        self.idx = idx
        self.name = name  # discuss this var

    def __hash__(self) -> int:
        return self.idx

    def __str__(self) -> str:
        return self.name

    def dual(self):
        return self.model.pi(self)

    pi = property(dual)


class LinExpr:

    def __init__(self, variables: List["Var"] = None,
                 coeffs: List[float] = None,
                 const: float = 0,
                 sense: str = ""):
        self.const: int = const
        self.expr: Dict[Var, float] = {}
        self.sense = sense

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

    def __iadd__(self, other):
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
        result = []

        if self.expr:
            for var, coeff in self.expr.items():
                result.append("+ " if coeff >= 0 else "- ")
                result.append(str(abs(coeff)) if abs(coeff) != 1 else "")
                result.append("{var} ".format(**locals()))

        if self.sense:
            result.append(self.sense + "= ")
            result.append(str(abs(self.const)) if self.const < 0 else "+ " + str(abs(self.const)))
        elif self.const != 0:
            result.append("+ " + str(abs(self.const)) if self.const > 0 else "- " + str(abs(self.const)))

        return "".join(result)

    def __eq__(self, other) -> "LinExpr":
        result = self - other
        result.sense = "="
        return result

    def __le__(self, other) -> "LinExpr":
        result = self - other
        result.sense = "<"
        return result

    def __ge__(self, other) -> "LinExpr":
        result = self - other
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
        copy = LinExpr()
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
        self.solver: Solver = None
        self.vars = []
        self.constrs = []

        # todo: implement code to detect solver automatically
        if solver_name == GUROBI:
            from milppy.gurobi import SolverGurobi
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

    def add_var(self,
                name: str = "",
                lb: float = 0.0,
                ub: float = INF,
                obj: float = 0.0,
                type: str = CONTINUOUS,
                column: "Column" = None
                ) -> "Var":
        idx = self.solver.add_var(obj, lb, ub, type, column, name)
        self.vars.append(Var(self, idx, name))
        return self.vars[-1]

    def add_constr(self, lin_expr: "LinExpr",
                   name: str = "") -> Constr:
        if isinstance(lin_expr, bool):
            return None  # empty constraint
        idx = self.solver.add_constr(lin_expr, name)
        self.constrs.append(Constr(self, idx, name))
        return self.constrs[-1]

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

    def x(self, var: "Var") -> float:
        return self.solver.x(var)


class Solver:

    def __init__(self, name: str, sense: str):
        self.name = name
        self.sense = sense

    def __del__(self): pass

    def add_var(self, name: str = "",
                obj: float = 0,
                lb: float = 0,
                ub: float = INF,
                type: str = CONTINUOUS,
                column: "Column" = None) -> int: pass

    def add_constr(self, lin_expr: "LinExpr", name: str = "") -> int: pass

    def copy(self) -> "Solver": pass

    def optimize(self) -> int: pass

    def pi(self, constr: "Constr") -> float: pass

    def set_start(self, variables: List["Var"], values: List[float]): pass

    def set_objective(self, lin_expr: "LinExpr", sense: str = ""): pass

    def write(self, file_path: str): pass

    def x(self, var: "Var") -> float: pass


class Var:

    def __init__(self, model: Model,
                 idx: int,
                 name: str = ""):
        self.model: Model = model
        self.idx: int = idx
        self.name = name  # discuss this var

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

    def value(self) -> float:
        return self.model.x(self)

    x = property(value)


def xsum(terms) -> LinExpr:
    result = LinExpr()
    for term in terms:
        result.add_term(term)
    return result


# function aliases
quicksum = xsum

# vim: ts=4 sw=4 et
