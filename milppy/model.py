from enum import Enum
from typing import Dict, List

# epsilon number (practical zero)
EPS = 10e-6

# infinity representation
INF = float("inf")

# optimization status
OPTIMAL = 0
INFEASIBLE = 1
UNBOUNDED = 2
FEASIBLE = 3
INT_INFEASIBLE = 4
NO_SOLUTION_FOUND = 5
ERROR = 6

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


class Column:

    def __init__(self,
                 constrs: List["Constr"] = None,
                 coeffs: List[float] = None):
        self.constrs = constrs if constrs else []
        self.coeffs = coeffs if coeffs else []


class Constr:

    def __init__(self,
                 model: "Model",
                 idx: int):
        self.model = model
        self.idx = idx

    def __hash__(self) -> int:
        return self.idx


class LinExpr:

    def __init__(self,
                 coeffs: List[float] = [],
                 variables: List["Var"] = [],
                 const: float = 0,
                 sense: str = ""):
        self.const: int = const
        self.expr: Dict[Var, float] = {}
        self.sense = sense

        for i in range(len(coeffs)):
            if coeffs[i] == 0:
                continue
            self.add_var(variables[i], coeffs[i])

    def __add__(self, other) -> "LinExpr":
        result: LinExpr = self.copy()
        if type(other) is Var:
            result.add_var(other, 1)
        elif type(other) is LinExpr:
            result.add_expr(other)
        elif type(other) is float or type(other) is int:
            result.add_const(other)
        return result

    def __radd__(self, other) -> "LinExpr":
        return self.__add__(other)

    def __sub__(self, other) -> "LinExpr":
        result: LinExpr = self.copy()
        if type(other) is Var:
            result.add_var(other, -1)
        elif type(other) is LinExpr:
            result.add_expr(-other)
        elif type(other) is float or type(other) is int:
            result.add_const(-other)
        return result

    def __rsub__(self, other) -> "LinExpr":
        return self.__add__(-other)

    def __mul__(self, other) -> "LinExpr":
        result: LinExpr = self.copy()
        assert type(other) is int or type(other) is float
        for var in result.expr.keys():
            result.expr[var] *= other

    def __rmul__(self, other) -> "LinExpr":
        return self.__mul__(other)

    def __str__(self) -> "LinExpr":
        result = ""
        for var, coeff in self.expr.items():
            result += "+ " if coeff >= 0 else "- "
            result += str(abs(coeff)) if abs(coeff) != 1 else ""
            result += "{var} ".format(**locals())
        if self.sense:
            result += self.sense + "= "
            result += str(abs(self.const)) if self.const < 0 else "+ " + str(abs(self.const))
        elif self.const != 0:
            result += "+ " + str(abs(self.const)) if self.const > 0 else "- " + str(abs(self.const))
        return result

    def __le__(self, other) -> "LinExpr":
        result: LinExpr = self.copy()
        result.sense = "<"
        return result - other

    def __ge__(self, other) -> "LinExpr":
        result: LinExpr = self.copy()
        result.sense = ">"
        return result - other

    def __eq__(self, other) -> "LinExpr":
        result: LinExpr = self.copy()
        result.sense = "="
        return result - other

    def add_const(self,
                  const: float):
        self.const += const

    def add_expr(self,
                 expr: "LinExpr"):
        for var, coeff in expr.expr.items():
            self.add_var(var, coeff)

    def add_term(self,
                 expr,
                 coeff: float = None):
        if type(expr) is Var:
            self.add_var(expr, coeff)
        elif type(expr) is LinExpr:
            self.add_expr(self, expr)
        elif type(expr) is float or type(expr) is int:
            self.add_const(expr)

    def add_var(self,
                var: "Var",
                coeff: float):
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

    def __init__(self,
                 name: str = "",
                 solver_name: str = GUROBI,
                 sense: str = MINIMIZE):
        # initializing variables with default values
        self.name: str = name
        self.solver: Solver = None
        self.sense: str = sense
        self.vars = []
        self.constrs = []

        # todo: implement code to detect solver automatically
        if solver_name == GUROBI:
            from milppy.gurobi import SolverGurobi
            self.solver = SolverGurobi(name, sense)

    def add_var(self,
                obj: float = 0,
                lb: float = 0,
                ub: float = INF,
                column: "Column" = None,
                type: str = CONTINUOUS,
                name: str = "") -> "Var":
        idx = self.solver.add_var(obj, lb, ub, column, type, name)
        self.vars.append(Var(self, idx))
        return self.vars[-1]

    def add_constr(self,
                   lin_expr: "LinExpr",
                   name: str = "") -> Constr:
        idx = self.solver.add_constr(lin_expr, name)
        self.constrs.append(Constr(self, idx))
        return self.constrs[-1]

    def optimize(self):
        self.solver.optimize()

    def set_obj(self, lin_expr: "LinExpr"):
        self.solver.set_obj(lin_expr)

    def write(self, path: str):
        self.solver.write(path)


class Solver:

    def __init__(self, name: str, sense: str):
        self.name = name
        self.sense = sense

    def add_var(self,
                obj: float = 0,
                lb: float = 0,
                ub: float = INF,
                column: "Column" = None,
                type: str = CONTINUOUS,
                name: str = "") -> int: pass

    def add_constr(self, lin_expr: "LinExpr", name: str = "") -> int: pass

    def optimize(self) -> int: pass

    def set_obj(self, lin_expr: "LinExpr"): pass

    def write(self, file_path: str): pass


class Var:

    def __init__(self,
                 model: Model,
                 idx: int):
        self.model: Model = model
        self.idx: int = idx

    def __hash__(self) -> int:
        return self.idx

    def __add__(self, other) -> LinExpr:
        if type(other) is Var:
            return LinExpr([1, 1], [self, other])
        elif type(other) is LinExpr:
            return other.__add__(self)
        elif type(other) is int or type(other) is float:
            return LinExpr([1], [self], other)

    def __radd__(self, other) -> LinExpr:
        return self.__add__(other)

    def __div__(self, other) -> LinExpr:
        assert type(other) is int or type(other) is float
        return self.__mul__(1.0 / other)

    def __mul__(self, other) -> LinExpr:
        assert type(other) is int or type(other) is float
        return LinExpr([other], [self])

    def __rmul__(self, other) -> LinExpr:
        return self.__mul__(other)

    def __sub__(self, other) -> LinExpr:
        if type(other) is Var:
            return LinExpr([1, -1], [self, other])
        elif type(other) is LinExpr:
            return other.__rsub__(self)
        elif type(other) is int or type(other) is float:
            return LinExpr([1], [self], -other)

    def __rsub__(self, other) -> LinExpr:
        if type(other) is Var:
            return LinExpr([-1, 1], [self, other])
        elif type(other) is LinExpr:
            return other.__sub__(self)
        elif type(other) is int or type(other) is float:
            return LinExpr([-1], [self], other)

    def __str__(self) -> LinExpr:
        return "x{self.idx}".format(**locals())

    def __le__(self, other) -> LinExpr:
        return LinExpr([1], [self], -1 * other, sense="<")

    def __ge__(self, other) -> LinExpr:
        return LinExpr([1], [self], -1 * other, sense=">")

    def __eq__(self, other) -> LinExpr:
        return LinExpr([1], [self], -1 * other, sense="=")
