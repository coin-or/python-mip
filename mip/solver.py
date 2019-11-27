"""This module implements the solver intependent communication layer of
Python-MIP
"""
from math import inf
from typing import List, Tuple
from mip.constants import INF, CONTINUOUS
from mip.constants import SearchEmphasis, OptimizationStatus


class Solver:
    """The solver is an abstract class with the solver independent
    API to communicate with the solver engine"""

    def __init__(self, model: "Model", name: str = '', sense: str = ''):
        self.model = model
        if name:
            self.name = name
        if sense:
            self.sense = sense

    def __del__(self):
        pass

    def add_var(self,
                name: str = "",
                obj: float = 0,
                lb: float = 0,
                ub: float = INF,
                var_type: str = CONTINUOUS,
                column: "Column" = None):
        pass

    def add_constr(self, lin_expr: "LinExpr", name: str = ""):
        pass

    def add_lazy_constr(self, lin_expr: "LinExpr"):
        pass

    def add_sos(self, sos: List[Tuple["Var", float]], sos_type: int):
        pass

    def add_cut(self, lin_expr: "LinExpr"):
        pass

    def get_objective_bound(self) -> float:
        pass

    def get_objective(self) -> "LinExpr":
        pass

    def get_objective_const(self) -> float:
        pass

    def relax(self):
        pass

    def optimize(self) -> OptimizationStatus:
        pass

    def get_objective_value(self) -> float:
        pass

    def get_log(self) -> List[Tuple[float, Tuple[float, float]]]:
        return []

    def get_objective_value_i(self, i: int) -> float:
        pass

    def get_num_solutions(self) -> int:
        pass

    def get_objective_sense(self) -> str:
        pass

    def set_objective_sense(self, sense: str):
        pass

    def set_start(self, start: List[Tuple["Var", float]]):
        pass

    def set_objective(self, lin_expr: "LinExpr", sense: str = ""):
        pass

    def set_objective_const(self, const: float):
        pass

    def set_callbacks(self,
                      branch_selector: "BranchSelector" = None,
                      incumbent_updater: "IncumbentUpdater" = None,
                      lazy_constrs_generator: "LazyConstrsGenerator" = None):
        pass

    def set_processing_limits(self,
                              max_time: float = inf,
                              max_nodes: int = inf,
                              max_sol: int = inf):
        pass

    def get_max_seconds(self) -> float:
        pass

    def set_max_seconds(self, max_seconds: float):
        pass

    def get_max_solutions(self) -> int:
        pass

    def set_max_solutions(self, max_solutions: int):
        pass

    def get_pump_passes(self) -> int:
        pass

    def set_pump_passes(self, passes: int):
        pass

    def get_max_nodes(self) -> int:
        pass

    def set_max_nodes(self, max_nodes: int):
        pass

    def set_num_threads(self, threads: int):
        pass

    def write(self, file_path: str):
        pass

    def read(self, file_path: str):
        pass

    def num_cols(self) -> int:
        pass

    def num_rows(self) -> int:
        pass

    def num_nz(self) -> int:
        pass

    def num_int(self) -> int:
        pass

    def get_emphasis(self) -> SearchEmphasis:
        pass

    def set_emphasis(self, emph: SearchEmphasis):
        pass

    def get_cutoff(self) -> float:
        pass

    def set_cutoff(self, cutoff: float):
        pass

    def get_mip_gap_abs(self) -> float:
        pass

    def set_mip_gap_abs(self, mip_gap_abs: float):
        pass

    def get_mip_gap(self) -> float:
        pass

    def set_mip_gap(self, mip_gap: float):
        pass

    def get_verbose(self) -> int:
        pass

    def set_verbose(self, verbose: int):
        pass

    # Constraint-related getters/setters

    def constr_get_expr(self, constr: "Constr") -> "LinExpr":
        pass

    def constr_set_expr(self, constr: "Constr", value: "LinExpr") -> "LinExpr":
        pass

    def constr_get_rhs(self, idx: int) -> float:
        pass

    def constr_set_rhs(self, idx: int, rhs: float):
        pass

    def constr_get_name(self, idx: int) -> str:
        pass

    def constr_get_pi(self, constr: "Constr") -> float:
        pass

    def constr_get_slack(self, constr: "Constr") -> float:
        pass

    def remove_constrs(self, constrsList: List[int]):
        pass

    def constr_get_index(self, name: str) -> int:
        pass

    # Variable-related getters/setters

    def var_get_lb(self, var: "Var") -> float:
        pass

    def var_set_lb(self, var: "Var", value: float):
        pass

    def var_get_ub(self, var: "Var") -> float:
        pass

    def var_set_ub(self, var: "Var", value: float):
        pass

    def var_get_obj(self, var: "Var") -> float:
        pass

    def var_set_obj(self, var: "Var", value: float):
        pass

    def var_get_var_type(self, var: "Var") -> str:
        pass

    def var_set_var_type(self, var: "Var", value: str):
        pass

    def var_get_column(self, var: "Var") -> "Column":
        pass

    def var_set_column(self, var: "Var", value: "Column"):
        pass

    def var_get_rc(self, var: "Var") -> float:
        pass

    def var_get_x(self, var: "Var") -> float:
        pass

    def var_get_xi(self, var: "Var", i: int) -> float:
        pass

    def var_get_name(self, idx: int) -> str:
        pass

    def remove_vars(self, varsList: List[int]):
        pass

    def var_get_index(self, name: str) -> int:
        pass

    def get_problem_name(self) -> str:
        pass

    def set_problem_name(self, name: str):
        pass

    def get_status(self) -> OptimizationStatus:
        pass
