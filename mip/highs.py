"Python-MIP interface to the HiGHS solver."

import glob
import numbers
import logging
import os
import os.path
import sys
from typing import List, Optional, Tuple, Union

import cffi

import mip

logger = logging.getLogger(__name__)

# try loading the solver library
ffi = cffi.FFI()
try:
    # first try user-defined path, if given
    ENV_KEY = "PMIP_HIGHS_LIBRARY"
    if ENV_KEY in os.environ:
        libfile = os.environ[ENV_KEY]
        logger.debug("Choosing HiGHS library {libfile} via {ENV_KEY}.")
    else:
        # try library shipped with highspy packaged
        import highspy

        pkg_path = os.path.dirname(highspy.__file__)

        # need library matching operating system
        if "linux" in sys.platform.lower():
            pattern = "highs_bindings.*.so"
        else:
            raise NotImplementedError(f"{sys.platform} not supported!")

        # there should only be one match
        [libfile] = glob.glob(os.path.join(pkg_path, pattern))
        logger.debug("Choosing HiGHS library {libfile} via highspy package.")

    highslib = ffi.dlopen(libfile)
    has_highs = True
except Exception as e:
    logger.error(f"An error occurred while loading the HiGHS library:\n{e}")
    has_highs = False

HEADER = """
typedef int HighsInt;

const HighsInt kHighsObjSenseMinimize = 1;
const HighsInt kHighsObjSenseMaximize = -1;

const HighsInt kHighsVarTypeContinuous = 0;
const HighsInt kHighsVarTypeInteger = 1;

void* Highs_create(void);
void Highs_destroy(void* highs);
HighsInt Highs_readModel(void* highs, const char* filename);
HighsInt Highs_writeModel(void* highs, const char* filename);
HighsInt Highs_run(void* highs);
HighsInt Highs_getModelStatus(const void* highs);
double Highs_getObjectiveValue(const void* highs);
HighsInt Highs_addVar(void* highs, const double lower, const double upper);
HighsInt Highs_addRow(
    void* highs, const double lower, const double upper, const HighsInt num_new_nz,
    const HighsInt* index, const double* value
);
HighsInt Highs_changeObjectiveSense(void* highs, const HighsInt sense);
HighsInt Highs_changeColIntegrality(
    void* highs, const HighsInt col, const HighsInt integrality
);
HighsInt Highs_changeColCost(void* highs, const HighsInt col, const double cost);
HighsInt Highs_changeColBounds(
    void* highs, const HighsInt col, const double lower, const double upper
);
HighsInt Highs_getRowsByRange(
    const void* highs, const HighsInt from_row, const HighsInt to_row,
    HighsInt* num_row, double* lower, double* upper, HighsInt* num_nz,
    HighsInt* matrix_start, HighsInt* matrix_index, double* matrix_value
);
HighsInt Highs_getNumCol(const void* highs);
"""

if has_highs:
    ffi.cdef(HEADER)


class SolverHighs(mip.Solver):
    def __init__(self, model: mip.Model, name: str, sense: str):
        if not has_highs:
            raise FileNotFoundError(
                "HiGHS not found."
                "Please install the `highspy` package, or"
                "set the `PMIP_HIGHS_LIBRARY` environment variable."
            )

        # Store reference to library so that it's not garbage-collected (when we
        # just use highslib in __del__, it had already become None)?!
        self._lib = highslib

        super().__init__(model, name, sense)

        # Model creation and initialization.
        self._model = highslib.Highs_create()

        sense_map = {
            mip.MAXIMIZE: self._lib.kHighsObjSenseMaximize,
            mip.MINIMIZE: self._lib.kHighsObjSenseMinimize,
        }
        status = self._lib.Highs_changeObjectiveSense(self._model, sense_map[sense])
        # TODO: handle status (everywhere)

        # Store additional data here, if HiGHS can't do it.
        self._name = name
        self._var_name: List[str] = []
        self._var_col: Dict[str, int] = {}

    def __del__(self):
        self._lib.Highs_destroy(self._model)

    def add_var(
        self: "SolverHighs",
        obj: numbers.Real = 0,
        lb: numbers.Real = 0,
        ub: numbers.Real = mip.INF,
        var_type: str = mip.CONTINUOUS,
        column: "Column" = None,
        name: str = "",
    ):
        # TODO: handle column data
        col: int = self._lib.Highs_getNumCol(self._model)
        status = self._lib.Highs_addVar(self._model, lb, ub)
        status = self._lib.Highs_changeColCost(self._model, col, obj)
        if var_type != mip.CONTINUOUS:
            status = self._lib.Highs_changeColIntegrality(
                self._model, col, self._lib.kHighsVarTypeInteger
            )

        # store name
        self._var_name.append(name)
        self._var_col[name] = col

    def add_constr(self: "SolverHighs", lin_expr: "mip.LinExpr", name: str = ""):
        pass

    def add_lazy_constr(self: "SolverHighs", lin_expr: "mip.LinExpr"):
        raise NotImplementedError()

    def add_sos(
        self: "SolverHighs",
        sos: List[Tuple["mip.Var", numbers.Real]],
        sos_type: int,
    ):
        raise NotImplementedError()

    def add_cut(self: "SolverHighs", lin_expr: "mip.LinExpr"):
        raise NotImplementedError()

    def get_objective_bound(self: "SolverHighs") -> numbers.Real:
        pass

    def get_objective(self: "SolverHighs") -> "mip.LinExpr":
        pass

    def get_objective_const(self: "SolverHighs") -> numbers.Real:
        pass

    def relax(self: "SolverHighs"):
        pass

    def generate_cuts(
        self,
        cut_types: Optional[List[mip.CutType]] = None,
        depth: int = 0,
        npass: int = 0,
        max_cuts: int = mip.INT_MAX,
        min_viol: numbers.Real = 1e-4,
    ) -> "mip.CutPool":
        raise NotImplementedError()

    def clique_merge(self, constrs: Optional[List["mip.Constr"]] = None):
        raise NotImplementedError()

    def optimize(
        self: "SolverHighs",
        relax: bool = False,
    ) -> "mip.OptimizationStatus":
        pass

    def get_objective_value(self: "SolverHighs") -> numbers.Real:
        pass

    def get_log(
        self: "SolverHighs",
    ) -> List[Tuple[numbers.Real, Tuple[numbers.Real, numbers.Real]]]:
        return []

    def get_objective_value_i(self: "SolverHighs", i: int) -> numbers.Real:
        pass

    def get_num_solutions(self: "SolverHighs") -> int:
        pass

    def get_objective_sense(self: "SolverHighs") -> str:
        pass

    def set_objective_sense(self: "SolverHighs", sense: str):
        pass

    def set_start(self: "SolverHighs", start: List[Tuple["mip.Var", numbers.Real]]):
        raise NotImplementedError()

    def set_objective(self: "SolverHighs", lin_expr: "mip.LinExpr", sense: str = ""):
        pass

    def set_objective_const(self: "SolverHighs", const: numbers.Real):
        pass

    def set_processing_limits(
        self: "SolverHighs",
        max_time: numbers.Real = mip.INF,
        max_nodes: int = mip.INT_MAX,
        max_sol: int = mip.INT_MAX,
        max_seconds_same_incumbent: float = mip.INF,
        max_nodes_same_incumbent: int = mip.INT_MAX,
    ):
        pass

    def get_max_seconds(self: "SolverHighs") -> numbers.Real:
        pass

    def set_max_seconds(self: "SolverHighs", max_seconds: numbers.Real):
        pass

    def get_max_solutions(self: "SolverHighs") -> int:
        pass

    def set_max_solutions(self: "SolverHighs", max_solutions: int):
        pass

    def get_pump_passes(self: "SolverHighs") -> int:
        raise NotImplementedError()

    def set_pump_passes(self: "SolverHighs", passes: int):
        raise NotImplementedError()

    def get_max_nodes(self: "SolverHighs") -> int:
        pass

    def set_max_nodes(self: "SolverHighs", max_nodes: int):
        pass

    def set_num_threads(self: "SolverHighs", threads: int):
        pass

    def write(self: "SolverHighs", file_path: str):
        pass

    def read(self: "SolverHighs", file_path: str):
        pass

    def num_cols(self: "SolverHighs") -> int:
        pass

    def num_rows(self: "SolverHighs") -> int:
        pass

    def num_nz(self: "SolverHighs") -> int:
        pass

    def num_int(self: "SolverHighs") -> int:
        pass

    def get_emphasis(self: "SolverHighs") -> mip.SearchEmphasis:
        pass

    def set_emphasis(self: "SolverHighs", emph: mip.SearchEmphasis):
        pass

    def get_cutoff(self: "SolverHighs") -> numbers.Real:
        pass

    def set_cutoff(self: "SolverHighs", cutoff: numbers.Real):
        pass

    def get_mip_gap_abs(self: "SolverHighs") -> numbers.Real:
        pass

    def set_mip_gap_abs(self: "SolverHighs", mip_gap_abs: numbers.Real):
        pass

    def get_mip_gap(self: "SolverHighs") -> numbers.Real:
        pass

    def set_mip_gap(self: "SolverHighs", mip_gap: numbers.Real):
        pass

    def get_verbose(self: "SolverHighs") -> int:
        pass

    def set_verbose(self: "SolverHighs", verbose: int):
        pass

    # Constraint-related getters/setters

    def constr_get_expr(self: "SolverHighs", constr: "mip.Constr") -> "mip.LinExpr":
        pass

    def constr_set_expr(
        self: "SolverHighs", constr: "mip.Constr", value: "mip.LinExpr"
    ) -> "mip.LinExpr":
        pass

    def constr_get_rhs(self: "SolverHighs", idx: int) -> numbers.Real:
        pass

    def constr_set_rhs(self: "SolverHighs", idx: int, rhs: numbers.Real):
        pass

    def constr_get_name(self: "SolverHighs", idx: int) -> str:
        pass

    def constr_get_pi(self: "SolverHighs", constr: "mip.Constr") -> numbers.Real:
        pass

    def constr_get_slack(self: "SolverHighs", constr: "mip.Constr") -> numbers.Real:
        pass

    def remove_constrs(self: "SolverHighs", constrsList: List[int]):
        pass

    def constr_get_index(self: "SolverHighs", name: str) -> int:
        pass

    # Variable-related getters/setters

    def var_get_branch_priority(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        raise NotImplementedError()

    def var_set_branch_priority(
        self: "SolverHighs", var: "mip.Var", value: numbers.Real
    ):
        raise NotImplementedError()

    def var_get_lb(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        pass

    def var_set_lb(self: "SolverHighs", var: "mip.Var", value: numbers.Real):
        pass

    def var_get_ub(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        pass

    def var_set_ub(self: "SolverHighs", var: "mip.Var", value: numbers.Real):
        pass

    def var_get_obj(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        pass

    def var_set_obj(self: "SolverHighs", var: "mip.Var", value: numbers.Real):
        pass

    def var_get_var_type(self: "SolverHighs", var: "mip.Var") -> str:
        pass

    def var_set_var_type(self: "SolverHighs", var: "mip.Var", value: str):
        pass

    def var_get_column(self: "SolverHighs", var: "mip.Var") -> "Column":
        pass

    def var_set_column(self: "SolverHighs", var: "mip.Var", value: "Column"):
        pass

    def var_get_rc(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        pass

    def var_get_x(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        """Assumes that the solution is available (should be checked
        before calling it"""

    def var_get_xi(self: "SolverHighs", var: "mip.Var", i: int) -> numbers.Real:
        pass

    def var_get_name(self: "SolverHighs", idx: int) -> str:
        pass

    def remove_vars(self: "SolverHighs", varsList: List[int]):
        pass

    def var_get_index(self: "SolverHighs", name: str) -> int:
        pass

    def get_problem_name(self: "SolverHighs") -> str:
        return self._name

    def set_problem_name(self: "SolverHighs", name: str):
        self._name = name

    def get_status(self: "SolverHighs") -> mip.OptimizationStatus:
        pass

    def cgraph_density(self: "SolverHighs") -> float:
        """Density of the conflict graph"""
        raise NotImplementedError()

    def conflicting(
        self: "SolverHighs",
        e1: Union["mip.LinExpr", "mip.Var"],
        e2: Union["mip.LinExpr", "mip.Var"],
    ) -> bool:
        """Checks if two assignment to binary variables are in conflict,
        returns none if no conflict graph is available"""
        raise NotImplementedError()

    def conflicting_nodes(
        self: "SolverHighs", v1: Union["mip.Var", "mip.LinExpr"]
    ) -> Tuple[List["mip.Var"], List["mip.Var"]]:
        """Returns all assignment conflicting with the assignment in v1 in the
        conflict graph.
        """
        raise NotImplementedError()

    def feature_values(self: "SolverHighs") -> List[float]:
        raise NotImplementedError()

    def feature_names(self: "SolverHighs") -> List[str]:
        raise NotImplementedError()
