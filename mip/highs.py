"Python-MIP interface to the HiGHS solver."

import glob
import numbers
import logging
import os
import os.path
import sys
from typing import Dict, List, Optional, Tuple, Union

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
        platform = sys.platform.lower()
        if "linux" in platform:
            pattern = "highs_bindings.*.so"
        elif platform.startswith("win"):
            pattern = "highs_bindings.*.pyd"
        elif any(platform.startswith(p) for p in ("darwin", "macos")):
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

const HighsInt kHighsStatusError = -1;
const HighsInt kHighsStatusOk = 0;
const HighsInt kHighsStatusWarning = 1;

const HighsInt kHighsObjSenseMinimize = 1;
const HighsInt kHighsObjSenseMaximize = -1;

const HighsInt kHighsVarTypeContinuous = 0;
const HighsInt kHighsVarTypeInteger = 1;

const HighsInt kHighsSolutionStatusNone = 0;
const HighsInt kHighsSolutionStatusInfeasible = 1;
const HighsInt kHighsSolutionStatusFeasible = 2;

const HighsInt kHighsModelStatusNotset = 0;
const HighsInt kHighsModelStatusLoadError = 1;
const HighsInt kHighsModelStatusModelError = 2;
const HighsInt kHighsModelStatusPresolveError = 3;
const HighsInt kHighsModelStatusSolveError = 4;
const HighsInt kHighsModelStatusPostsolveError = 5;
const HighsInt kHighsModelStatusModelEmpty = 6;
const HighsInt kHighsModelStatusOptimal = 7;
const HighsInt kHighsModelStatusInfeasible = 8;
const HighsInt kHighsModelStatusUnboundedOrInfeasible = 9;
const HighsInt kHighsModelStatusUnbounded = 10;
const HighsInt kHighsModelStatusObjectiveBound = 11;
const HighsInt kHighsModelStatusObjectiveTarget = 12;
const HighsInt kHighsModelStatusTimeLimit = 13;
const HighsInt kHighsModelStatusIterationLimit = 14;
const HighsInt kHighsModelStatusUnknown = 15;
const HighsInt kHighsModelStatusSolutionLimit = 16;

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
HighsInt Highs_changeObjectiveOffset(void* highs, const double offset);
HighsInt Highs_changeObjectiveSense(void* highs, const HighsInt sense);
HighsInt Highs_changeColIntegrality(
    void* highs, const HighsInt col, const HighsInt integrality
);
HighsInt Highs_changeColsIntegralityByRange(
    void* highs, const HighsInt from_col, const HighsInt to_col,
    const HighsInt* integrality
);
HighsInt Highs_changeColCost(void* highs, const HighsInt col, const double cost);
HighsInt Highs_changeColBounds(
    void* highs, const HighsInt col, const double lower, const double upper
);
HighsInt Highs_changeCoeff(
    void* highs, const HighsInt row, const HighsInt col, const double value
);
HighsInt Highs_getRowsByRange(
    const void* highs, const HighsInt from_row, const HighsInt to_row,
    HighsInt* num_row, double* lower, double* upper, HighsInt* num_nz,
    HighsInt* matrix_start, HighsInt* matrix_index, double* matrix_value
);
HighsInt Highs_getColsByRange(
    const void* highs, const HighsInt from_col, const HighsInt to_col,
    HighsInt* num_col, double* costs, double* lower, double* upper,
    HighsInt* num_nz, HighsInt* matrix_start, HighsInt* matrix_index,
    double* matrix_value
);
HighsInt Highs_getObjectiveOffset(const void* highs, double* offset);
HighsInt Highs_getObjectiveSense(const void* highs, HighsInt* sense);
HighsInt Highs_getNumCol(const void* highs);
HighsInt Highs_getNumRow(const void* highs);
HighsInt Highs_getNumNz(const void* highs);
HighsInt Highs_getDoubleInfoValue(
    const void* highs, const char* info, double* value
);
HighsInt Highs_getIntInfoValue(
    const void* highs, const char* info, int* value
);
HighsInt Highs_getIntOptionValue(
    const void* highs, const char* option, HighsInt* value
);
HighsInt Highs_getDoubleOptionValue(
    const void* highs, const char* option, double* value
);
HighsInt Highs_getBoolOptionValue(
    const void* highs, const char* option, bool* value
);
HighsInt Highs_setIntOptionValue(
    void* highs, const char* option, const HighsInt value
);
HighsInt Highs_setDoubleOptionValue(
    void* highs, const char* option, const double value
);
HighsInt Highs_setBoolOptionValue(
    void* highs, const char* option, const bool value
);
HighsInt Highs_getSolution(
    const void* highs, double* col_value, double* col_dual,
    double* row_value, double* row_dual
);
HighsInt Highs_deleteRowsBySet(
    void* highs, const HighsInt num_set_entries, const HighsInt* set
);
HighsInt Highs_deleteColsBySet(
    void* highs, const HighsInt num_set_entries, const HighsInt* set
);
"""

if has_highs:
    ffi.cdef(HEADER)

    STATUS_ERROR = highslib.kHighsStatusError

    def check(status):
        "Check return status and raise error if not OK."
        if status == STATUS_ERROR:
            raise mip.InterfacingError("Unknown error in call to HiGHS.")


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
        self.set_objective_sense(sense)

        # Store additional data here, if HiGHS can't do it.
        self._name: str = name
        self._var_name: List[str] = []
        self._var_col: Dict[str, int] = {}
        self._var_type: List[str] = []
        self._cons_name: List[str] = []
        self._cons_col: Dict[str, int] = {}

        # Also store solution (when available)
        self._x = []
        self._rc = []
        self._pi = []

    def __del__(self):
        self._lib.Highs_destroy(self._model)

    def _get_int_info_value(self: "SolverHighs", name: str) -> int:
        value = ffi.new("int*")
        check(self._lib.Highs_getIntInfoValue(self._model, name.encode("UTF-8"), value))
        return value[0]

    def _get_double_info_value(self: "SolverHighs", name: str) -> float:
        value = ffi.new("double*")
        check(
            self._lib.Highs_getDoubleInfoValue(self._model, name.encode("UTF-8"), value)
        )
        return value[0]

    def _get_int_option_value(self: "SolverHighs", name: str) -> int:
        value = ffi.new("int*")
        check(
            self._lib.Highs_getIntOptionValue(self._model, name.encode("UTF-8"), value)
        )
        return value[0]

    def _get_double_option_value(self: "SolverHighs", name: str) -> float:
        value = ffi.new("double*")
        check(
            self._lib.Highs_getDoubleOptionValue(
                self._model, name.encode("UTF-8"), value
            )
        )
        return value[0]

    def _get_bool_option_value(self: "SolverHighs", name: str) -> float:
        value = ffi.new("bool*")
        check(
            self._lib.Highs_getBoolOptionValue(self._model, name.encode("UTF-8"), value)
        )
        return value[0]

    def _set_int_option_value(self: "SolverHighs", name: str, value: int):
        check(
            self._lib.Highs_setIntOptionValue(self._model, name.encode("UTF-8"), value)
        )

    def _set_double_option_value(self: "SolverHighs", name: str, value: float):
        check(
            self._lib.Highs_setDoubleOptionValue(
                self._model, name.encode("UTF-8"), value
            )
        )

    def _set_bool_option_value(self: "SolverHighs", name: str, value: float):
        check(
            self._lib.Highs_setBoolOptionValue(self._model, name.encode("UTF-8"), value)
        )

    def _change_coef(self: "SolverHighs", row: int, col: int, value: float):
        "Overwrite a single coefficient in the matrix."
        check(self._lib.Highs_changeCoeff(self._model, row, col, value))

    def _set_column(self: "SolverHighs", col: int, column: "mip.Column"):
        "Overwrite coefficients of one column."
        # We also have to set to 0 all coefficients of the old column, so we
        # fetch that first.
        var = self.model.vars[col]
        old_column = self.var_get_column(var)
        coeffs = {cons.idx: 0.0 for cons in old_column.constrs}
        coeffs.update(
            {cons.idx: coef for cons, coef in zip(column.constrs, column.coeffs)}
        )
        for row, coef in coeffs.items():
            self._change_coef(row, col, coef)

    def add_var(
        self: "SolverHighs",
        obj: numbers.Real = 0,
        lb: numbers.Real = 0,
        ub: numbers.Real = mip.INF,
        var_type: str = mip.CONTINUOUS,
        column: "mip.Column" = None,
        name: str = "",
    ):
        col: int = self.num_cols()
        check(self._lib.Highs_addVar(self._model, lb, ub))
        check(self._lib.Highs_changeColCost(self._model, col, obj))
        if var_type != mip.CONTINUOUS:
            check(
                self._lib.Highs_changeColIntegrality(
                    self._model, col, self._lib.kHighsVarTypeInteger
                )
            )

        if column:
            self._set_column(col, column)

        # store name & type
        self._var_name.append(name)
        self._var_col[name] = col
        self._var_type.append(var_type)

    def add_constr(self: "SolverHighs", lin_expr: "mip.LinExpr", name: str = ""):
        row: int = self.num_rows()

        # equation expressed as two-sided inequality
        lower = -lin_expr.const
        upper = -lin_expr.const
        if lin_expr.sense == mip.LESS_OR_EQUAL:
            lower = -mip.INF
        elif lin_expr.sense == mip.GREATER_OR_EQUAL:
            upper = mip.INF
        else:
            assert lin_expr.sense == mip.EQUAL

        num_new_nz = len(lin_expr.expr)
        index = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
        value = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])

        check(
            self._lib.Highs_addRow(self._model, lower, upper, num_new_nz, index, value)
        )

        # store name
        self._cons_name.append(name)
        self._cons_col[name] = row

    def add_lazy_constr(self: "SolverHighs", lin_expr: "mip.LinExpr"):
        raise NotImplementedError("HiGHS doesn't support lazy constraints!")

    def add_sos(
        self: "SolverHighs",
        sos: List[Tuple["mip.Var", numbers.Real]],
        sos_type: int,
    ):
        raise NotImplementedError("HiGHS doesn't support SOS!")

    def add_cut(self: "SolverHighs", lin_expr: "mip.LinExpr"):
        raise NotImplementedError("HiGHS doesn't support cut callbacks!")

    def get_objective_bound(self: "SolverHighs") -> numbers.Real:
        return self._get_double_info_value("mip_dual_bound")

    def get_objective(self: "SolverHighs") -> "mip.LinExpr":
        n = self.num_cols()
        num_col = ffi.new("int*")
        costs = ffi.new("double[]", n)
        lower = ffi.new("double[]", n)
        upper = ffi.new("double[]", n)
        num_nz = ffi.new("int*")
        check(
            self._lib.Highs_getColsByRange(
                self._model,
                0,  # from_col
                n - 1,  # to_col
                num_col,
                costs,
                lower,
                upper,
                num_nz,
                ffi.NULL,  # matrix_start
                ffi.NULL,  # matrix_index
                ffi.NULL,  # matrix_value
            )
        )
        obj_expr = mip.xsum(
            costs[i] * self.model.vars[i] for i in range(n) if costs[i] != 0.0
        )
        obj_expr.add_const(self.get_objective_const())
        obj_expr.sense = self.get_objective_sense()
        return obj_expr

    def get_objective_const(self: "SolverHighs") -> numbers.Real:
        offset = ffi.new("double*")
        check(self._lib.Highs_getObjectiveOffset(self._model, offset))
        return offset[0]

    def _all_cols_continuous(self: "SolverHighs"):
        n = self.num_cols()
        integrality = ffi.new(
            "int[]", [self._lib.kHighsVarTypeContinuous for i in range(n)]
        )
        check(
            self._lib.Highs_changeColsIntegralityByRange(
                self._model, 0, n - 1, integrality
            )
        )

    def _reset_var_types(self: "SolverHighs"):
        var_type_map = {
            mip.CONTINUOUS: self._lib.kHighsVarTypeContinuous,
            mip.BINARY: self._lib.kHighsVarTypeInteger,
            mip.INTEGER: self._lib.kHighsVarTypeInteger,
        }
        integrality = ffi.new("int[]", [var_type_map[vt] for vt in self._var_type])
        n = self.num_cols()
        check(
            self._lib.Highs_changeColsIntegralityByRange(
                self._model, 0, n - 1, integrality
            )
        )

    def relax(self: "SolverHighs"):
        # change integrality of all columns
        self._all_cols_continuous()
        self._var_type = [mip.CONTINUOUS] * len(self._var_type)

    def generate_cuts(
        self,
        cut_types: Optional[List[mip.CutType]] = None,
        depth: int = 0,
        npass: int = 0,
        max_cuts: int = mip.INT_MAX,
        min_viol: numbers.Real = 1e-4,
    ) -> "mip.CutPool":
        raise NotImplementedError("HiGHS doesn't support manual cut generation.")

    def clique_merge(self, constrs: Optional[List["mip.Constr"]] = None):
        raise NotImplementedError("HiGHS doesn't support clique merging!")

    def optimize(
        self: "SolverHighs",
        relax: bool = False,
    ) -> "mip.OptimizationStatus":
        if relax:
            # Temporarily change variable types. Original types are still stored
            # in self._var_type.
            self._all_cols_continuous()

        check(self._lib.Highs_run(self._model))

        # store solution values for later access
        opt_status = self.get_status()
        if opt_status in (
            mip.OptimizationStatus.OPTIMAL,
            mip.OptimizationStatus.FEASIBLE,
        ):
            n, m = self.num_cols(), self.num_rows()
            col_value = ffi.new("double[]", n)
            col_dual = ffi.new("double[]", n)
            row_value = ffi.new("double[]", m)
            row_dual = ffi.new("double[]", m)
            check(
                self._lib.Highs_getSolution(
                    self._model, col_value, col_dual, row_value, row_dual
                )
            )
            self._x = [col_value[j] for j in range(n)]
            self._rc = [col_dual[j] for j in range(n)]

            if self._has_dual_solution():
                self._pi = [row_dual[i] for i in range(m)]

        if relax:
            # Undo the temporary changes.
            self._reset_var_types()

        return opt_status

    def get_objective_value(self: "SolverHighs") -> numbers.Real:
        # only give value if we have stored a solution
        if self._x:
            return self._lib.Highs_getObjectiveValue(self._model)

    def get_log(
        self: "SolverHighs",
    ) -> List[Tuple[numbers.Real, Tuple[numbers.Real, numbers.Real]]]:
        raise NotImplementedError("HiGHS doesn't give access to a progress log.")

    def get_objective_value_i(self: "SolverHighs", i: int) -> numbers.Real:
        raise NotImplementedError("HiGHS doesn't store multiple solutions.")

    def get_num_solutions(self: "SolverHighs") -> int:
        # Multiple solutions are not supported (through C API?).
        return 1 if self._has_primal_solution() else 0

    def get_objective_sense(self: "SolverHighs") -> str:
        sense = ffi.new("int*")
        check(self._lib.Highs_getObjectiveSense(self._model, sense))
        sense_map = {
            self._lib.kHighsObjSenseMaximize: mip.MAXIMIZE,
            self._lib.kHighsObjSenseMinimize: mip.MINIMIZE,
        }
        return sense_map[sense[0]]

    def set_objective_sense(self: "SolverHighs", sense: str):
        sense_map = {
            mip.MAXIMIZE: self._lib.kHighsObjSenseMaximize,
            mip.MINIMIZE: self._lib.kHighsObjSenseMinimize,
        }
        check(self._lib.Highs_changeObjectiveSense(self._model, sense_map[sense]))

    def set_start(self: "SolverHighs", start: List[Tuple["mip.Var", numbers.Real]]):
        raise NotImplementedError("HiGHS doesn't support a start solution.")

    def set_objective(self: "SolverHighs", lin_expr: "mip.LinExpr", sense: str = ""):
        # set coefficients
        for var, coef in lin_expr.expr.items():
            check(self._lib.Highs_changeColCost(self._model, var.idx, coef))

        self.set_objective_const(lin_expr.const)
        if lin_expr.sense:
            self.set_objective_sense(lin_expr.sense)

    def set_objective_const(self: "SolverHighs", const: numbers.Real):
        check(self._lib.Highs_changeObjectiveOffset(self._model, const))

    def set_processing_limits(
        self: "SolverHighs",
        max_time: numbers.Real = mip.INF,
        max_nodes: int = mip.INT_MAX,
        max_sol: int = mip.INT_MAX,
        max_seconds_same_incumbent: float = mip.INF,
        max_nodes_same_incumbent: int = mip.INT_MAX,
    ):
        if max_time != mip.INF:
            self.set_max_seconds(max_time)
        if max_nodes != mip.INT_MAX:
            self.set_max_nodes(max_nodes)
        if max_sol != mip.INT_MAX:
            self.set_max_solutions(max_sol)
        if max_seconds_same_incumbent != mip.INF:
            raise NotImplementedError("Can't set max_seconds_same_incumbent!")
        if max_nodes_same_incumbent != mip.INT_MAX:
            self.set_max_nodes_same_incumbent(max_nodes_same_incumbent)

    def get_max_seconds(self: "SolverHighs") -> numbers.Real:
        return self._get_double_option_value("time_limit")

    def set_max_seconds(self: "SolverHighs", max_seconds: numbers.Real):
        self._set_double_option_value("time_limit", max_seconds)

    def get_max_solutions(self: "SolverHighs") -> int:
        return self._get_int_option_value("mip_max_improving_sols")

    def set_max_solutions(self: "SolverHighs", max_solutions: int):
        self._get_int_option_value("mip_max_improving_sols", max_solutions)

    def get_pump_passes(self: "SolverHighs") -> int:
        raise NotImplementedError("HiGHS doesn't support pump passes.")

    def set_pump_passes(self: "SolverHighs", passes: int):
        raise NotImplementedError("HiGHS doesn't support pump passes.")

    def get_max_nodes(self: "SolverHighs") -> int:
        return self._get_int_option_value("mip_max_nodes")

    def set_max_nodes(self: "SolverHighs", max_nodes: int):
        self._set_int_option_value("mip_max_nodes", max_nodes)

    def get_max_nodes_same_incumbent(self: "SolverHighs") -> int:
        return self._get_int_option_value("mip_max_stall_nodes")

    def set_max_nodes_same_incumbent(self: "SolverHighs", max_nodes_same_incumbent: int):
        self._set_int_option_value("mip_max_stall_nodes", max_nodes_same_incumbent)

    def set_num_threads(self: "SolverHighs", threads: int):
        self._set_int_option_value("threads", threads)

    def write(self: "SolverHighs", file_path: str):
        check(self._lib.Highs_writeModel(self._model, file_path.encode("utf-8")))

    def read(self: "SolverHighs", file_path: str):
        check(self._lib.Highs_readModel(self._model, file_path.encode("utf-8")))

    def num_cols(self: "SolverHighs") -> int:
        return self._lib.Highs_getNumCol(self._model)

    def num_rows(self: "SolverHighs") -> int:
        return self._lib.Highs_getNumRow(self._model)

    def num_nz(self: "SolverHighs") -> int:
        return self._lib.Highs_getNumNz(self._model)

    def num_int(self: "SolverHighs") -> int:
        return sum(vt != mip.CONTINUOUS for vt in self._var_type)

    def get_emphasis(self: "SolverHighs") -> mip.SearchEmphasis:
        raise NotImplementedError("HiGHS doesn't support search emphasis.")

    def set_emphasis(self: "SolverHighs", emph: mip.SearchEmphasis):
        raise NotImplementedError("HiGHS doesn't support search emphasis.")

    def get_cutoff(self: "SolverHighs") -> numbers.Real:
        return self._get_double_option_value("objective_bound")

    def set_cutoff(self: "SolverHighs", cutoff: numbers.Real):
        self._set_double_option_value("objective_bound", cutoff)

    def get_mip_gap_abs(self: "SolverHighs") -> numbers.Real:
        return self._get_double_option_value("mip_abs_gap")

    def set_mip_gap_abs(self: "SolverHighs", mip_gap_abs: numbers.Real):
        self._set_double_option_value("mip_abs_gap", mip_gap_abs)

    def get_mip_gap(self: "SolverHighs") -> numbers.Real:
        return self._get_double_option_value("mip_rel_gap")

    def set_mip_gap(self: "SolverHighs", mip_gap: numbers.Real):
        self._set_double_option_value("mip_rel_gap", mip_gap)

    def get_verbose(self: "SolverHighs") -> int:
        return self._get_bool_option_value("output_flag")

    def set_verbose(self: "SolverHighs", verbose: int):
        self._set_bool_option_value("output_flag", verbose)

    # Constraint-related getters/setters

    def constr_get_expr(self: "SolverHighs", constr: "mip.Constr") -> "mip.LinExpr":
        row = constr.idx
        # Call method twice:
        #  - first, to get the sizes for coefficients,
        num_row = ffi.new("int*")
        lower = ffi.new("double[]", 1)
        upper = ffi.new("double[]", 1)
        num_nz = ffi.new("int*")
        # TODO: We also pass a non-NULL matrix_start, which should not be
        # needed, but works around a known bug in HiGHS' C API.
        _tmp_matrix_start = ffi.new("int[]", 1)
        check(
            self._lib.Highs_getRowsByRange(
                self._model,
                row,
                row,
                num_row,
                lower,
                upper,
                num_nz,
                _tmp_matrix_start,
                ffi.NULL,
                ffi.NULL,
            )
        )

        #  - second, to get the coefficients in pre-allocated arrays.
        if num_nz[0] == 0:
            # early exit for empty expressions
            expr = mip.xsum([])
        else:
            matrix_start = ffi.new("int[]", 1)
            matrix_index = ffi.new("int[]", num_nz[0])
            matrix_value = ffi.new("double[]", num_nz[0])
            check(
                self._lib.Highs_getRowsByRange(
                    self._model,
                    row,
                    row,
                    num_row,
                    lower,
                    upper,
                    num_nz,
                    matrix_start,
                    matrix_index,
                    matrix_value,
                )
            )
            expr = mip.xsum(
                matrix_value[i] * self.model.vars[matrix_index[i]]
                for i in range(num_nz[0])
            )

        # Also set sense and constant
        lhs, rhs = lower[0], upper[0]
        if rhs < mip.INF:
            expr -= rhs
            if lhs > -mip.INF:
                assert lhs == rhs
                expr.sense = mip.EQUAL
            else:
                expr.sense = mip.LESS_OR_EQUAL
        else:
            if lhs > -mip.INF:
                expr -= lhs
                expr.sense = mip.GREATER_OR_EQUAL
            else:
                raise ValueError("Unbounded constraint?!")
        return expr

    def constr_set_expr(
        self: "SolverHighs", constr: "mip.Constr", value: "mip.LinExpr"
    ) -> "mip.LinExpr":
        # We also have to set to 0 all coefficients of the old row, so we
        # fetch that first.
        coeffs = {var: 0.0 for var in constr.expr}

        # Then we fetch the new coefficients and overwrite.
        coeffs.update(value.expr.items())

        # Finally, we change the coeffs in HiGHS' matrix one-by-one.
        for var, coef in coeffs.items():
            self._change_coef(constr.idx, var.idx, coef)

    def constr_get_rhs(self: "SolverHighs", idx: int) -> numbers.Real:
        # fetch both lower and upper bound
        num_row = ffi.new("int*")
        lower = ffi.new("double[]", 1)
        upper = ffi.new("double[]", 1)
        num_nz = ffi.new("int*")
        check(
            self._lib.Highs_getRowsByRange(
                self._model,
                idx,
                idx,
                num_row,
                lower,
                upper,
                num_nz,
                ffi.NULL,
                ffi.NULL,
                ffi.NULL,
            )
        )

        # case distinction for sense
        if lower[0] == -mip.INF:
            return -upper[0]
        if upper[0] == mip.INF:
            return -lower[0]
        assert lower[0] == upper[0]
        return -lower[0]

    def constr_set_rhs(self: "SolverHighs", idx: int, rhs: numbers.Real):
        # first need to figure out which bound to change (lower or upper)
        num_row = ffi.new("int*")
        lower = ffi.new("double[]", 1)
        upper = ffi.new("double[]", 1)
        num_nz = ffi.new("int*")
        check(
            self._lib.Highs_getRowsByRange(
                self._model,
                idx,
                idx,
                num_row,
                lower,
                upper,
                num_nz,
                ffi.NULL,
                ffi.NULL,
                ffi.NULL,
            )
        )

        # update bounds as needed
        lb, ub = lower[0], upper[0]
        if lb != -mip.INF:
            lb = -rhs
        if ub != mip.INF:
            ub = -rhs

        # set new bounds
        check(self._lib.Highs_changeRowBounds(self._model, idx, lb, ub))

    def constr_get_name(self: "SolverHighs", idx: int) -> str:
        return self._cons_name(idx)

    def constr_get_pi(self: "SolverHighs", constr: "mip.Constr") -> numbers.Real:
        if self._pi:
            return self._pi[constr.idx]

    def constr_get_slack(self: "SolverHighs", constr: "mip.Constr") -> numbers.Real:
        expr = constr.expr
        activity = sum(coef * var.x for var, coef in expr.expr.items())
        rhs = -expr.const
        slack = rhs - activity
        if expr.sense == mip.LESS_OR_EQUAL:
            return slack
        elif expr.sense == mip.GREATER_OR_EQUAL:
            return -slack
        elif expr.sense == mip.EQUAL:
            return -abs(slack)
        else:
            raise ValueError(f"Invalid constraint sense: {expr.sense}")

    def remove_constrs(self: "SolverHighs", constrsList: List[int]):
        set_ = ffi.new("int[]", constrsList)
        check(self._lib.Highs_deleteRowsBySet(self._model, len(constrsList), set_))

    def constr_get_index(self: "SolverHighs", name: str) -> int:
        return self._cons_col(name)

    # Variable-related getters/setters

    def var_get_branch_priority(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        # TODO: Is actually not supported by HiGHS, but we mimic the behavior of
        # CBC and simply pretend that it's always 0.
        return 0

    def var_set_branch_priority(
        self: "SolverHighs", var: "mip.Var", value: numbers.Real
    ):
        # TODO: better raise warning/error instead?
        pass

    def var_get_lb(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        num_col = ffi.new("int*")
        costs = ffi.new("double[]", 1)
        lower = ffi.new("double[]", 1)
        upper = ffi.new("double[]", 1)
        num_nz = ffi.new("int*")
        check(
            self._lib.Highs_getColsByRange(
                self._model,
                var.idx,  # from_col
                var.idx,  # to_col
                num_col,
                costs,
                lower,
                upper,
                num_nz,
                ffi.NULL,  # matrix_start
                ffi.NULL,  # matrix_index
                ffi.NULL,  # matrix_value
            )
        )
        return lower[0]

    def var_set_lb(self: "SolverHighs", var: "mip.Var", value: numbers.Real):
        # can only set both bounds, so we just set the old upper bound
        old_upper = self.var_get_ub(var)
        check(self._lib.Highs_changeColBounds(self._model, var.idx, value, old_upper))

    def var_get_ub(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        num_col = ffi.new("int*")
        costs = ffi.new("double[]", 1)
        lower = ffi.new("double[]", 1)
        upper = ffi.new("double[]", 1)
        num_nz = ffi.new("int*")
        check(
            self._lib.Highs_getColsByRange(
                self._model,
                var.idx,  # from_col
                var.idx,  # to_col
                num_col,
                costs,
                lower,
                upper,
                num_nz,
                ffi.NULL,  # matrix_start
                ffi.NULL,  # matrix_index
                ffi.NULL,  # matrix_value
            )
        )
        return upper[0]

    def var_set_ub(self: "SolverHighs", var: "mip.Var", value: numbers.Real):
        # can only set both bounds, so we just set the old lower bound
        old_lower = self.var_get_lb(var)
        check(self._lib.Highs_changeColBounds(self._model, var.idx, old_lower, value))

    def var_get_obj(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        num_col = ffi.new("int*")
        costs = ffi.new("double[]", 1)
        lower = ffi.new("double[]", 1)
        upper = ffi.new("double[]", 1)
        num_nz = ffi.new("int*")
        check(
            self._lib.Highs_getColsByRange(
                self._model,
                var.idx,  # from_col
                var.idx,  # to_col
                num_col,
                costs,
                lower,
                upper,
                num_nz,
                ffi.NULL,  # matrix_start
                ffi.NULL,  # matrix_index
                ffi.NULL,  # matrix_value
            )
        )
        return costs[0]

    def var_set_obj(self: "SolverHighs", var: "mip.Var", value: numbers.Real):
        check(self._lib.Highs_changeColCost(self._model, var.idx, value))

    def var_get_var_type(self: "SolverHighs", var: "mip.Var") -> str:
        return self._var_type[var.idx]

    def var_set_var_type(self: "SolverHighs", var: "mip.Var", value: str):
        if value != mip.CONTINUOUS:
            check(
                self._lib.Highs_changeColIntegrality(
                    self._model, var.idx, self._lib.kHighsVarTypeInteger
                )
            )
        self._var_type[var.idx] = value

    def var_get_column(self: "SolverHighs", var: "mip.Var") -> "mip.Column":
        # Call method twice:
        #  - first, to get the sizes for coefficients,
        num_col = ffi.new("int*")
        costs = ffi.new("double[]", 1)
        lower = ffi.new("double[]", 1)
        upper = ffi.new("double[]", 1)
        num_nz = ffi.new("int*")
        check(
            self._lib.Highs_getColsByRange(
                self._model,
                var.idx,  # from_col
                var.idx,  # to_col
                num_col,
                costs,
                lower,
                upper,
                num_nz,
                ffi.NULL,  # matrix_start
                ffi.NULL,  # matrix_index
                ffi.NULL,  # matrix_value
            )
        )
        #  - second, to get the coefficients in pre-allocated arrays.
        matrix_start = ffi.new("int[]", 1)
        matrix_index = ffi.new("int[]", num_nz[0])
        matrix_value = ffi.new("double[]", num_nz[0])
        check(
            self._lib.Highs_getColsByRange(
                self._model,
                var.idx,  # from_col
                var.idx,  # to_col
                num_col,
                costs,
                lower,
                upper,
                num_nz,
                matrix_start,
                matrix_index,
                matrix_value,
            )
        )

        return mip.Column(
            constrs=[self.model.constrs[matrix_index[i]] for i in range(num_nz[0])],
            coeffs=[matrix_value[i] for i in range(num_nz[0])],
        )

    def var_set_column(self: "SolverHighs", var: "mip.Var", value: "mip.Column"):
        self._set_column(var.idx, value)

    def var_get_rc(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        if self._rc:
            return self._rc[var.idx]

    def var_get_x(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        if self._x:
            return self._x[var.idx]

    def var_get_xi(self: "SolverHighs", var: "mip.Var", i: int) -> numbers.Real:
        raise NotImplementedError("HiGHS doesn't store multiple solutions.")

    def var_get_name(self: "SolverHighs", idx: int) -> str:
        return self._var_name[idx]

    def remove_vars(self: "SolverHighs", varsList: List[int]):
        set_ = ffi.new("int[]", varsList)
        check(self._lib.Highs_deleteColsBySet(self._model, len(varsList), set_))

    def var_get_index(self: "SolverHighs", name: str) -> int:
        return self._var_col[name]

    def get_problem_name(self: "SolverHighs") -> str:
        return self._name

    def set_problem_name(self: "SolverHighs", name: str):
        self._name = name

    def _get_primal_solution_status(self: "SolverHighs"):
        return self._get_int_info_value("primal_solution_status")

    def _has_primal_solution(self: "SolverHighs"):
        return (
            self._get_primal_solution_status() == self._lib.kHighsSolutionStatusFeasible
        )

    def _get_dual_solution_status(self: "SolverHighs"):
        return self._get_int_info_value("dual_solution_status")

    def _has_dual_solution(self: "SolverHighs"):
        return self._get_dual_solution_status() == self._lib.kHighsSolutionStatusFeasible

    def get_status(self: "SolverHighs") -> mip.OptimizationStatus:
        OS = mip.OptimizationStatus
        status_map = {
            self._lib.kHighsModelStatusNotset: OS.OTHER,
            self._lib.kHighsModelStatusLoadError: OS.ERROR,
            self._lib.kHighsModelStatusModelError: OS.ERROR,
            self._lib.kHighsModelStatusPresolveError: OS.ERROR,
            self._lib.kHighsModelStatusSolveError: OS.ERROR,
            self._lib.kHighsModelStatusPostsolveError: OS.ERROR,
            self._lib.kHighsModelStatusModelEmpty: OS.OTHER,
            self._lib.kHighsModelStatusOptimal: OS.OPTIMAL,
            self._lib.kHighsModelStatusInfeasible: OS.INFEASIBLE,
            self._lib.kHighsModelStatusUnboundedOrInfeasible: OS.UNBOUNDED,
            # ... or should it be INFEASIBLE?
            self._lib.kHighsModelStatusUnbounded: OS.UNBOUNDED,
            self._lib.kHighsModelStatusObjectiveBound: None,
            self._lib.kHighsModelStatusObjectiveTarget: None,
            self._lib.kHighsModelStatusTimeLimit: None,
            self._lib.kHighsModelStatusIterationLimit: None,
            self._lib.kHighsModelStatusUnknown: OS.OTHER,
            self._lib.kHighsModelStatusSolutionLimit: None,
        }
        highs_status = self._lib.Highs_getModelStatus(self._model)
        status = status_map[highs_status]
        if status is None:
            # depends on solution status
            status = OS.FEASIBLE if self._has_primal_solution() else OS.NO_SOLUTION_FOUND
        return status

    def cgraph_density(self: "SolverHighs") -> float:
        """Density of the conflict graph"""
        raise NotImplementedError("HiGHS doesn't support conflict graph.")

    def conflicting(
        self: "SolverHighs",
        e1: Union["mip.LinExpr", "mip.Var"],
        e2: Union["mip.LinExpr", "mip.Var"],
    ) -> bool:
        """Checks if two assignment to binary variables are in conflict,
        returns none if no conflict graph is available"""
        raise NotImplementedError("HiGHS doesn't support conflict graph.")

    def conflicting_nodes(
        self: "SolverHighs", v1: Union["mip.Var", "mip.LinExpr"]
    ) -> Tuple[List["mip.Var"], List["mip.Var"]]:
        """Returns all assignment conflicting with the assignment in v1 in the
        conflict graph.
        """
        raise NotImplementedError("HiGHS doesn't support conflict graph.")

    def feature_values(self: "SolverHighs") -> List[float]:
        raise NotImplementedError("HiGHS doesn't support feature extraction.")

    def feature_names(self: "SolverHighs") -> List[str]:
        raise NotImplementedError("HiGHS doesn't support feature extraction.")
