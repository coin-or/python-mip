"""This module implements the solver intependent communication layer of
Python-MIP
"""
from typing import List, Tuple, Optional, Union
from sys import maxsize
import numbers
import mip


class Solver:
    """The solver is an abstract class with the solver independent
    API to communicate with the solver engine"""

    def __init__(self: "Solver", model: "Model", name: str = "", sense: str = ""):
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
        obj: numbers.Real = 0,
        lb: numbers.Real = 0,
        ub: numbers.Real = mip.INF,
        var_type: str = mip.CONTINUOUS,
        column: "Column" = None,
    ):
        pass

    def add_constr(self: "Solver", lin_expr: "mip.LinExpr", name: str = ""):
        pass

    def add_lazy_constr(self: "Solver", lin_expr: "mip.LinExpr"):
        pass

    def add_sos(
        self: "Solver", sos: List[Tuple["mip.Var", numbers.Real]], sos_type: int,
    ):
        pass

    def add_cut(self: "Solver", lin_expr: "mip.LinExpr"):
        pass

    def get_objective_bound(self: "Solver") -> numbers.Real:
        pass

    def get_objective(self: "Solver") -> "mip.LinExpr":
        pass

    def get_objective_const(self: "Solver") -> numbers.Real:
        pass

    def relax(self: "Solver"):
        pass

    def generate_cuts(
        self,
        cut_types: Optional[List[mip.CutType]] = None,
        max_cuts: int = maxsize,
        min_viol: numbers.Real = 1e-4,
    ) -> "mip.CutPool":
        pass

    def optimize(self: "Solver", relax: bool = False,) -> "mip.OptimizationStatus":
        pass

    def get_objective_value(self: "Solver") -> numbers.Real:
        pass

    def get_log(
        self: "Solver",
    ) -> List[Tuple[numbers.Real, Tuple[numbers.Real, numbers.Real]]]:
        return []

    def get_objective_value_i(self: "Solver", i: int) -> numbers.Real:
        pass

    def get_num_solutions(self: "Solver") -> int:
        pass

    def get_objective_sense(self: "Solver") -> str:
        pass

    def set_objective_sense(self: "Solver", sense: str):
        pass

    def set_start(self: "Solver", start: List[Tuple["mip.Var", numbers.Real]]):
        pass

    def set_objective(self: "Solver", lin_expr: "mip.LinExpr", sense: str = ""):
        pass

    def set_objective_const(self: "Solver", const: numbers.Real):
        pass

    def set_processing_limits(
        self: "Solver",
        max_time: numbers.Real = mip.INF,
        max_nodes: int = maxsize,
        max_sol: int = maxsize,
    ):
        pass

    def get_max_seconds(self: "Solver") -> numbers.Real:
        pass

    def set_max_seconds(self: "Solver", max_seconds: numbers.Real):
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

    def get_emphasis(self: "Solver") -> mip.SearchEmphasis:
        pass

    def set_emphasis(self: "Solver", emph: mip.SearchEmphasis):
        pass

    def get_cutoff(self: "Solver") -> numbers.Real:
        pass

    def set_cutoff(self: "Solver", cutoff: numbers.Real):
        pass

    def get_mip_gap_abs(self: "Solver") -> numbers.Real:
        pass

    def set_mip_gap_abs(self: "Solver", mip_gap_abs: numbers.Real):
        pass

    def get_mip_gap(self: "Solver") -> numbers.Real:
        pass

    def set_mip_gap(self: "Solver", mip_gap: numbers.Real):
        pass

    def get_verbose(self: "Solver") -> int:
        pass

    def set_verbose(self: "Solver", verbose: int):
        pass

    # Constraint-related getters/setters

    def constr_get_expr(self: "Solver", constr: "mip.Constr") -> "mip.LinExpr":
        pass

    def constr_set_expr(
        self: "Solver", constr: "mip.Constr", value: "mip.LinExpr"
    ) -> "mip.LinExpr":
        pass

    def constr_get_rhs(self: "Solver", idx: int) -> numbers.Real:
        pass

    def constr_set_rhs(self: "Solver", idx: int, rhs: numbers.Real):
        pass

    def constr_get_name(self: "Solver", idx: int) -> str:
        pass

    def constr_get_pi(self: "Solver", constr: "mip.Constr") -> numbers.Real:
        pass

    def constr_get_slack(self: "Solver", constr: "mip.Constr") -> numbers.Real:
        pass

    def remove_constrs(self: "Solver", constrsList: List[int]):
        pass

    def constr_get_index(self: "Solver", name: str) -> int:
        pass

    # Variable-related getters/setters

    def var_get_lb(self: "Solver", var: "mip.Var") -> numbers.Real:
        pass

    def var_set_lb(self: "Solver", var: "mip.Var", value: numbers.Real):
        pass

    def var_get_ub(self: "Solver", var: "mip.Var") -> numbers.Real:
        pass

    def var_set_ub(self: "Solver", var: "mip.Var", value: numbers.Real):
        pass

    def var_get_obj(self: "Solver", var: "mip.Var") -> numbers.Real:
        pass

    def var_set_obj(self: "Solver", var: "mip.Var", value: numbers.Real):
        pass

    def var_get_var_type(self: "Solver", var: "mip.Var") -> str:
        pass

    def var_set_var_type(self: "Solver", var: "mip.Var", value: str):
        pass

    def var_get_column(self: "Solver", var: "mip.Var") -> "Column":
        pass

    def var_set_column(self: "Solver", var: "mip.Var", value: "Column"):
        pass

    def var_get_rc(self: "Solver", var: "mip.Var") -> numbers.Real:
        pass

    def var_get_x(self: "Solver", var: "mip.Var") -> numbers.Real:
        """Assumes that the solution is available (should be checked
           before calling it"""

    def var_get_xi(self: "Solver", var: "mip.Var", i: int) -> numbers.Real:
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

    def get_status(self: "Solver") -> mip.OptimizationStatus:
        pass

    def cgraph_density(self: "Solver") -> float:
        """Density of the conflict graph"""
        pass

    def conflicting(
        self: "Solver",
        e1: Union["mip.LinExpr", "mip.Var"],
        e2: Union["mip.LinExpr", "mip.Var"],
    ) -> bool:
        """Checks if two assignment to binary variables are in conflict,
        returns none if no conflict graph is available"""
        pass

    def conflicting_nodes(
        self: "Solver", v1: Union["mip.Var", "mip.LinExpr"]
    ) -> Tuple[List["mip.Var"], List["mip.Var"]]:
        """Returns all assignment conflicting with the assignment in v1 in the
        conflict graph.
        """
        pass
