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
        self.name = name

    def __hash__(self) -> int:
        return self.idx

    def __str__(self) -> str:
        return self.name


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
        if type(other) is Var:
            result.add_var(other, 1)
        elif type(other) is LinExpr:
            result.add_expr(other)
        elif type(other) is float or type(other) is int:
            result.add_const(other)
        return result

    def __radd__(self, other) -> "LinExpr":
        return self.__add__(other)

    def __iadd__(self, other):
        if type(other) is Var:
            self.add_var(other, 1)
        elif type(other) is LinExpr:
            self.add_expr(other)
        elif type(other) is float or type(other) is int:
            self.add_const(other)
        return self

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

    def __isub__(self, other) -> "LinExpr":
        if type(other) is Var:
            self.add_var(other, -1)
        elif type(other) is LinExpr:
            self.add_expr(-other)
        elif type(other) is float or type(other) is int:
            self.add_const(-other)
        return self

    def __mul__(self, other) -> "LinExpr":
        result: LinExpr = self.copy()
        assert type(other) is int or type(other) is float
        for var in result.expr.keys():
            result.expr[var] *= other
        return result

    def __rmul__(self, other) -> "LinExpr":
        return self.__mul__(other)

    def __imul__(self, other) -> "LinExpr":
        assert type(other) is int or type(other) is float
        for var in self.expr.keys():
            self.expr[var] *= other
        return self

    def __truediv__(self, other) -> "LinExpr":
        result: LinExpr = self.copy()
        assert type(other) is int or type(other) is float
        for var in result.expr.keys():
            result.expr[var] /= other
        return result

    def __itruediv__(self, other) -> "LinExpr":
        assert type(other) is int or type(other) is float
        for var in self.expr.keys():
            self.expr[var] /= other
        return self

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
        if type(expr) is Var:
            self.add_var(expr, coeff)
        elif type(expr) is LinExpr:
            self.add_expr(expr, coeff)
        elif type(expr) is float or type(expr) is int:
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
                name : str = "",
                lb: float = 0.0,
                ub: float = INF,
                obj: float = 0.0,
                type: str = CONTINUOUS,
                column: "Column" = None
                ) -> "Var":
        idx = self.solver.add_var(obj, lb, ub, column, type, name)
        self.vars.append(Var(self, idx, name))
        return self.vars[-1]

    def add_constr(self, lin_expr: "LinExpr",
                   name: str = "") -> Constr:
        if type(lin_expr) is bool:
        	return  # empty constraint
        idx = self.solver.add_constr(lin_expr, name)
        self.constrs.append(Constr(self, idx, name))
        return self.constrs[-1]

    def optimize(self):
        self.solver.optimize()

    def set_start(self, variables: List["Var"], values: List[float]):
        self.solver.set_start(variables, values)

    def set_objective(self, expr, sense: str = ""):
        if type(expr) is int or type(expr) is float:
            self.solver.set_objective(LinExpr([], [], expr))
        elif type(expr) is Var:
            self.solver.set_objective(LinExpr([expr], [1]))
        elif type(expr) is LinExpr:
            self.solver.set_objective(expr, sense)

    def write(self, path: str):
        self.solver.write(path)



class Solver:

    def __init__(self, name: str, sense: str):
        self.name = name
        self.sense = sense

    def add_var(self, obj: float = 0,
                lb: float = 0,
                ub: float = INF,
                column: "Column" = None,
                type: str = CONTINUOUS,
                name: str = "") -> int: pass

    def add_constr(self, lin_expr: "LinExpr", name: str = "") -> int: pass

    def optimize(self) -> int: pass

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
        self.name = name

    def __hash__(self) -> int:
        return self.idx

    def __add__(self, other) -> LinExpr:
        if type(other) is Var:
            return LinExpr([self, other], [1, 1])
        elif type(other) is LinExpr:
            return other.__add__(self)
        elif type(other) is int or type(other) is float:
            return LinExpr([self], [1], other)

    def __radd__(self, other) -> LinExpr:
        return self.__add__(other)

    def __sub__(self, other) -> LinExpr:
        if type(other) is Var:
            return LinExpr([self, other], [1, -1])
        elif type(other) is LinExpr:
            return other.__rsub__(self)
        elif type(other) is int or type(other) is float:
            return LinExpr([self], [1], -other)

    def __rsub__(self, other) -> LinExpr:
        if type(other) is Var:
            return LinExpr([self, other], [-1, 1])
        elif type(other) is LinExpr:
            return other.__sub__(self)
        elif type(other) is int or type(other) is float:
            return LinExpr([self], [-1], other)

    def __mul__(self, other) -> LinExpr:
        assert type(other) is int or type(other) is float
        return LinExpr([self], [other])

    def __rmul__(self, other) -> LinExpr:
        return self.__mul__(other)

    def __truediv__(self, other) -> LinExpr:
        assert type(other) is int or type(other) is float
        return self.__mul__(1.0 / other)

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
        return self.model.solver.x(self)


def xsum(terms) -> LinExpr:
    result = LinExpr()
    for term in terms:
        result.add_term(term)
    return result


# function aliases
quicksum = xsum

# vim: ts=4 sw=4 et
