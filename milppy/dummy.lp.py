from milppy.model import *
from ctypes import *
from ctypes.util import *


class DummyConstr:
    pass


class DummyVar:
    def __init__(self, name: str,
                 lb: float,
                 ub: float,
                 type: str):
        self.name = name
        self.lb = lb
        self.ub = ub
        self.type = type


class SolverDummy(Solver):

    def __init__(self, name: str, sense: str):
        super().__init__(name, sense)

    def add_var(self, name: str = "", obj: float = 0, lb: float = 0, ub: float = INF, type: str = CONTINUOUS,
                column: "Column" = None) -> int:
        return super().add_var(name, obj, lb, ub, type, column)

    def add_constr(self, lin_expr: "LinExpr", name: str = "") -> int:
        return super().add_constr(lin_expr, name)

    def optimize(self) -> int:
        return super().optimize()

    def pi(self, constr: "Constr") -> float:
        return super().pi(constr)

    def set_start(self, variables: List["Var"], values: List[float]):
        super().set_start(variables, values)

    def set_objective(self, lin_expr: "LinExpr", sense: str = ""):
        super().set_objective(lin_expr, sense)

    def write(self, file_path: str):
        super().write(file_path)

    def x(self, var: "Var") -> float:
        return super().x(var)
