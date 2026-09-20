"""This module implements the solver intependent communication layer of
Python-MIP
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import mip


class Solver(ABC):
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

    @abstractmethod
    def add_var(
        self: "Solver",
        obj: mip.Numeric = 0,
        lb: mip.Numeric = 0,
        ub: mip.Numeric = mip.INF,
        var_type: str = mip.CONTINUOUS,
        column: "Column | None" = None,
        name: str = "",
    ):
        ...

    @abstractmethod
    def add_constr(self: "Solver", lin_expr: "mip.LinExpr", name: str = ""):
        ...

    @abstractmethod
    def add_lazy_constr(self: "Solver", lin_expr: "mip.LinExpr"):
        ...

    def add_sos(
        self: "Solver",
        sos: list[tuple["mip.Var", mip.Numeric]],
        sos_type: int,
    ):
        pass

    @abstractmethod
    def add_cut(self: "Solver", lin_expr: "mip.LinExpr"):
        ...

    @abstractmethod
    def get_objective_bound(self: "Solver") -> mip.Numeric:
        ...

    @abstractmethod
    def get_objective(self: "Solver") -> "mip.LinExpr":
        ...

    @abstractmethod
    def get_objective_const(self: "Solver") -> mip.Numeric:
        ...

    @abstractmethod
    def relax(self: "Solver"):
        ...

    def generate_cuts(
        self,
        cut_types: list[mip.CutType] | None = None,
        depth: int = 0,
        npass: int = 0,
        max_cuts: int = mip.INT_MAX,
        min_viol: mip.Numeric = 1e-4,
    ) -> "mip.CutPool":
        pass

    def clique_merge(self, constrs: list["mip.Constr"] | None = None):
        pass

    @abstractmethod
    def optimize(
        self: "Solver",
        relax: bool = False,
        lp_preprocess: bool = False,
    ) -> "mip.OptimizationStatus":
        ...

    @abstractmethod
    def get_objective_value(self: "Solver") -> mip.Numeric | None:
        ...

    def get_log(
        self: "Solver",
    ) -> list[tuple[mip.Numeric, tuple[mip.Numeric, mip.Numeric]]]:
        return []

    @abstractmethod
    def get_objective_value_i(self: "Solver", i: int) -> mip.Numeric:
        ...

    @abstractmethod
    def get_num_solutions(self: "Solver") -> int:
        ...

    @abstractmethod
    def get_objective_sense(self: "Solver") -> str:
        ...

    @abstractmethod
    def set_objective_sense(self: "Solver", sense: str):
        ...

    @abstractmethod
    def set_start(self: "Solver", start: list[tuple["mip.Var", mip.Numeric]]):
        ...

    @abstractmethod
    def set_objective(self: "Solver", lin_expr: "mip.LinExpr", sense: str = ""):
        ...

    def set_objective_const(self: "Solver", const: mip.Numeric):
        pass

    @abstractmethod
    def set_processing_limits(
        self: "Solver",
        max_time: mip.Numeric = mip.INF,
        max_nodes: int = mip.INT_MAX,
        max_sol: int = mip.INT_MAX,
        max_seconds_same_incumbent: float = mip.INF,
        max_nodes_same_incumbent: int = mip.INT_MAX,
    ):
        ...

    @abstractmethod
    def get_max_seconds(self: "Solver") -> mip.Numeric:
        ...

    @abstractmethod
    def set_max_seconds(self: "Solver", max_seconds: mip.Numeric):
        ...

    @abstractmethod
    def get_max_solutions(self: "Solver") -> int:
        ...

    @abstractmethod
    def set_max_solutions(self: "Solver", max_solutions: int):
        ...

    @abstractmethod
    def get_pump_passes(self: "Solver") -> int:
        ...

    @abstractmethod
    def set_pump_passes(self: "Solver", passes: int):
        ...

    @abstractmethod
    def get_max_nodes(self: "Solver") -> int:
        ...

    @abstractmethod
    def set_max_nodes(self: "Solver", max_nodes: int):
        ...

    def get_max_iter(self: "Solver") -> int:
        pass

    def set_max_iter(self: "Solver", max_iter: int):
        pass

    @abstractmethod
    def set_num_threads(self: "Solver", threads: int):
        ...

    @abstractmethod
    def write(self: "Solver", file_path: str):
        ...

    @abstractmethod
    def read(self: "Solver", file_path: str):
        ...

    @abstractmethod
    def num_cols(self: "Solver") -> int:
        ...

    @abstractmethod
    def num_rows(self: "Solver") -> int:
        ...

    @abstractmethod
    def num_nz(self: "Solver") -> int:
        ...

    @abstractmethod
    def num_int(self: "Solver") -> int:
        ...

    @abstractmethod
    def get_emphasis(self: "Solver") -> mip.SearchEmphasis:
        ...

    @abstractmethod
    def set_emphasis(self: "Solver", emph: mip.SearchEmphasis):
        ...

    @abstractmethod
    def get_cutoff(self: "Solver") -> mip.Numeric:
        ...

    @abstractmethod
    def set_cutoff(self: "Solver", cutoff: mip.Numeric):
        ...

    @abstractmethod
    def get_mip_gap_abs(self: "Solver") -> mip.Numeric:
        ...

    @abstractmethod
    def set_mip_gap_abs(self: "Solver", mip_gap_abs: mip.Numeric):
        ...

    @abstractmethod
    def get_mip_gap(self: "Solver") -> mip.Numeric:
        ...

    @abstractmethod
    def set_mip_gap(self: "Solver", mip_gap: mip.Numeric):
        ...

    @abstractmethod
    def get_verbose(self: "Solver") -> int:
        ...

    @abstractmethod
    def set_verbose(self: "Solver", verbose: int):
        pass

    # Constraint-related getters/setters

    @abstractmethod
    def constr_get_expr(self: "Solver", constr: "mip.Constr") -> "mip.LinExpr":
        ...

    def constr_set_expr(
        self: "Solver", constr: "mip.Constr", value: "mip.LinExpr"
    ) -> None:
        pass

    def constr_get_rhs(self: "Solver", idx: int) -> mip.Numeric:
        pass

    def constr_set_rhs(self: "Solver", idx: int, rhs: mip.Numeric):
        pass

    @abstractmethod
    def constr_get_name(self: "Solver", idx: int) -> str:
        ...

    @abstractmethod
    def constr_get_pi(self: "Solver", constr: "mip.Constr") -> mip.Numeric | None:
        ...

    @abstractmethod
    def constr_get_slack(self: "Solver", constr: "mip.Constr") -> mip.Numeric:
        ...

    @abstractmethod
    def remove_constrs(self: "Solver", constrsList: list[int]):
        ...

    @abstractmethod
    def constr_get_index(self: "Solver", name: str) -> int:
        pass

    # Variable-related getters/setters

    @abstractmethod
    def var_get_branch_priority(self: "Solver", var: "mip.Var") -> mip.Numeric:
        ...

    def var_set_branch_priority(self: "Solver", var: "mip.Var", value: mip.Numeric):
        pass

    @abstractmethod
    def var_get_lb(self: "Solver", var: "mip.Var") -> mip.Numeric:
        ...

    @abstractmethod
    def var_set_lb(self: "Solver", var: "mip.Var", value: mip.Numeric):
        ...

    @abstractmethod
    def var_get_ub(self: "Solver", var: "mip.Var") -> mip.Numeric:
        ...

    @abstractmethod
    def var_set_ub(self: "Solver", var: "mip.Var", value: mip.Numeric):
        ...

    @abstractmethod
    def var_get_obj(self: "Solver", var: "mip.Var") -> mip.Numeric:
        ...

    @abstractmethod
    def var_set_obj(self: "Solver", var: "mip.Var", value: mip.Numeric):
        ...

    @abstractmethod
    def var_get_var_type(self: "Solver", var: "mip.Var") -> str:
        ...

    @abstractmethod
    def var_set_var_type(self: "Solver", var: "mip.Var", value: str):
        ...

    @abstractmethod
    def var_get_column(self: "Solver", var: "mip.Var") -> "Column":
        ...

    @abstractmethod
    def var_set_column(self: "Solver", var: "mip.Var", value: "Column"):
        ...

    @abstractmethod
    def var_get_rc(self: "Solver", var: "mip.Var") -> mip.Numeric | None:
        ...

    @abstractmethod
    def var_get_x(self: "Solver", var: "mip.Var") -> mip.Numeric | None:
        """Assumes that the solution is available (should be checked
        before calling it"""

    @abstractmethod
    def var_get_xi(self: "Solver", var: "mip.Var", i: int) -> mip.Numeric:
        ...

    @abstractmethod
    def var_get_name(self: "Solver", idx: int) -> str:
        ...

    @abstractmethod
    def remove_vars(self: "Solver", varsList: list[int]):
        ...

    @abstractmethod
    def var_get_index(self: "Solver", name: str) -> int:
        ...

    @abstractmethod
    def get_problem_name(self: "Solver") -> str:
        ...

    @abstractmethod
    def set_problem_name(self: "Solver", name: str):
        ...

    def get_status(self: "Solver") -> mip.OptimizationStatus:
        pass

    def cgraph_density(self: "Solver") -> float:
        """Density of the conflict graph"""
        pass

    def conflicting(
        self: "Solver",
        e1: "mip.LinExpr" | "mip.Var",
        e2: "mip.LinExpr" | "mip.Var",
    ) -> bool:
        """Checks if two assignment to binary variables are in conflict,
        returns none if no conflict graph is available"""
        pass

    def conflicting_nodes(
        self: "Solver", v1: "mip.Var" | "mip.LinExpr"
    ) -> tuple[list["mip.Var"], list["mip.Var"]]:
        """Returns all assignment conflicting with the assignment in v1 in the
        conflict graph.
        """
        pass

    def feature_values(self: "Solver") -> list[float]:
        pass

    def feature_names(self: "Solver") -> list[str]:
        pass
