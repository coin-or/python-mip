"""This module implements the solver intependent communication layer of
Python-MIP
"""
from typing import List, Tuple, TYPE_CHECKING, Optional
from sys import maxsize
from mip.constants import INF, CONTINUOUS
from mip.constants import SearchEmphasis, OptimizationStatus, CutType
from mip.callbacks import CutPool

if TYPE_CHECKING:
    from mip.model import Model
    from mip.entities import Column, Var, LinExpr, Constr


class Solver:
    """The solver is an abstract class with the solver independent
    API to communicate with the solver engine"""

    def __init__(
        self: "Solver", model: "Model", name: str = "", sense: str = ""
    ):
        self.model = model
        if name:
            self.name = name
        if sense:
            self.sense = sense

    def __del__(self: "Solver"):
        pass

    def add_var(
        self: "Solver",
        name: str = "",
        obj: float = 0,
        lb: float = 0,
        ub: float = INF,
        var_type: str = CONTINUOUS,
        column: "Column" = None,
    ):
        pass

    def add_constr(self: "Solver", lin_expr: "LinExpr", name: str = ""):
        pass

    def add_lazy_constr(self: "Solver", lin_expr: "LinExpr"):
        pass

    def add_sos(self: "Solver", sos: List[Tuple["Var", float]], sos_type: int):
        pass

    def add_cut(self: "Solver", lin_expr: "LinExpr"):
        pass

    def get_objective_bound(self: "Solver") -> float:
        pass

    def get_objective(self: "Solver") -> "LinExpr":
        pass

    def get_objective_const(self: "Solver") -> float:
        pass

    def relax(self: "Solver"):
        pass

    def generate_cuts(
        self,
        cut_types: Optional[List[CutType]] = None,
        max_cuts: int = maxsize,
        min_viol: float = 1e-4,
    ) -> CutPool:
        pass

    def optimize(self: "Solver", relax: bool = False,) -> OptimizationStatus:
        pass

    def get_objective_value(self: "Solver") -> float:
        pass

    def get_log(self: "Solver") -> List[Tuple[float, Tuple[float, float]]]:
        return []

    def get_objective_value_i(self: "Solver", i: int) -> float:
        pass

    def get_num_solutions(self: "Solver") -> int:
        pass

    def get_objective_sense(self: "Solver") -> str:
        pass

    def set_objective_sense(self: "Solver", sense: str):
        pass

    def set_start(self: "Solver", start: List[Tuple["Var", float]]):
        pass

    def set_objective(self: "Solver", lin_expr: "LinExpr", sense: str = ""):
        pass

    def set_objective_const(self: "Solver", const: float):
        pass

    def set_processing_limits(
        self: "Solver",
        max_time: float = INF,
        max_nodes: int = maxsize,
        max_sol: int = maxsize,
    ):
        pass

    def get_max_seconds(self: "Solver") -> float:
        pass

    def set_max_seconds(self: "Solver", max_seconds: float):
        pass

    def get_max_solutions(self: "Solver") -> int:
        pass

    def set_max_solutions(self: "Solver", max_solutions: int):
        pass

    def get_pump_passes(self: "Solver") -> int:
        pass

    def set_pump_passes(self: "Solver", passes: int):
        pass

    def get_max_nodes(self: "Solver") -> int:
        pass

    def set_max_nodes(self: "Solver", max_nodes: int):
        pass

    def set_num_threads(self: "Solver", threads: int):
        pass

    def write(self: "Solver", file_path: str):
        pass

    def read(self: "Solver", file_path: str):
        pass

    def num_cols(self: "Solver") -> int:
        pass

    def num_rows(self: "Solver") -> int:
        pass

    def num_nz(self: "Solver") -> int:
        pass

    def num_int(self: "Solver") -> int:
        pass

    def get_emphasis(self: "Solver") -> SearchEmphasis:
        pass

    def set_emphasis(self: "Solver", emph: SearchEmphasis):
        pass

    def get_cutoff(self: "Solver") -> float:
        pass

    def set_cutoff(self: "Solver", cutoff: float):
        pass

    def get_mip_gap_abs(self: "Solver") -> float:
        pass

    def set_mip_gap_abs(self: "Solver", mip_gap_abs: float):
        pass

    def get_mip_gap(self: "Solver") -> float:
        pass

    def set_mip_gap(self: "Solver", mip_gap: float):
        pass

    def get_verbose(self: "Solver") -> int:
        pass

    def set_verbose(self: "Solver", verbose: int):
        pass

    # Constraint-related getters/setters

    def constr_get_expr(self: "Solver", constr: "Constr") -> "LinExpr":
        pass

    def constr_set_expr(
        self: "Solver", constr: "Constr", value: "LinExpr"
    ) -> "LinExpr":
        pass

    def constr_get_rhs(self: "Solver", idx: int) -> float:
        pass

    def constr_set_rhs(self: "Solver", idx: int, rhs: float):
        pass

    def constr_get_name(self: "Solver", idx: int) -> str:
        pass

    def constr_get_pi(self: "Solver", constr: "Constr") -> float:
        pass

    def constr_get_slack(self: "Solver", constr: "Constr") -> float:
        pass

    def remove_constrs(self: "Solver", constrsList: List[int]):
        pass

    def constr_get_index(self: "Solver", name: str) -> int:
        pass

    # Variable-related getters/setters

    def var_get_lb(self: "Solver", var: "Var") -> float:
        pass

    def var_set_lb(self: "Solver", var: "Var", value: float):
        pass

    def var_get_ub(self: "Solver", var: "Var") -> float:
        pass

    def var_set_ub(self: "Solver", var: "Var", value: float):
        pass

    def var_get_obj(self: "Solver", var: "Var") -> float:
        pass

    def var_set_obj(self: "Solver", var: "Var", value: float):
        pass

    def var_get_var_type(self: "Solver", var: "Var") -> str:
        pass

    def var_set_var_type(self: "Solver", var: "Var", value: str):
        pass

    def var_get_column(self: "Solver", var: "Var") -> "Column":
        pass

    def var_set_column(self: "Solver", var: "Var", value: "Column"):
        pass

    def var_get_rc(self: "Solver", var: "Var") -> float:
        pass

    def var_get_x(self: "Solver", var: "Var") -> float:
        """Assumes that the solution is available (should be checked
           before calling it"""

    def var_get_xi(self: "Solver", var: "Var", i: int) -> float:
        pass

    def var_get_name(self: "Solver", idx: int) -> str:
        pass

    def remove_vars(self: "Solver", varsList: List[int]):
        pass

    def var_get_index(self: "Solver", name: str) -> int:
        pass

    def get_problem_name(self: "Solver") -> str:
        pass

    def set_problem_name(self: "Solver", name: str):
        pass

    def get_status(self: "Solver") -> OptimizationStatus:
        pass
