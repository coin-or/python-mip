import mip
from mip.model import Model, Solver, Var, Constr, Column, LinExpr
from mip.constants import MAXIMIZE, SearchEmphasis, CONTINUOUS, BINARY, \
    INTEGER, MINIMIZE, EQUAL, LESS_OR_EQUAL, GREATER_OR_EQUAL, \
    OptimizationStatus
from typing import Dict, List, Tuple
from sys import platform, maxsize
from os.path import dirname, isfile
import os
from cffi import FFI

warningMessages = 0

ffi = FFI()
CData = ffi.CData
has_cbc = False
os_is_64_bit = maxsize > 2**32
INF = float('inf')

# for variables and rows
MAX_NAME_SIZE = 512

try:
    pathmip = dirname(mip.__file__)
    pathlib = os.path.join(pathmip, 'libraries')
    libfile = ''
    if 'linux' in platform.lower():
        if os_is_64_bit:
            libfile = os.path.join(pathlib, 'cbc-c-linux-x86-64.so')
    elif platform.lower().startswith('win'):
        if os_is_64_bit:
            libfile = os.path.join(pathlib, 'cbc-c-windows-x86-64.dll')
        else:
            libfile = os.path.join(pathlib, 'cbc-c-windows-x86-32.dll')
    elif platform.lower().startswith('darwin'):
        if os_is_64_bit:
            libfile = os.path.join(pathlib, 'cbc-c-darwin-x86-64.a')
    if not libfile:
        raise Exception(
            "You operating system/platform is not supported")
    cbclib = ffi.dlopen(libfile)
    has_cbc = True
except Exception:
    has_cbc = False
    print('cbc not found')

if has_cbc:

    ffi.cdef("""
    typedef void(*cbc_cut_callback)(void *osiSolver,
        void *osiCuts, void *appdata);

    typedef int (*cbc_incumbent_callback)(void *cbcModel,
        double obj, int nz,
        char **vnames, double *x, void *appData);

    typedef void Cbc_Model;

    void *Cbc_newModel();

    void Cbc_readLp(Cbc_Model *model, const char *file);

    void Cbc_readMps(Cbc_Model *model, const char *file);

    void Cbc_writeLp(Cbc_Model *model, const char *file);

    void Cbc_writeMps(Cbc_Model *model, const char *file);

    int Cbc_getNumCols(Cbc_Model *model);

    int Cbc_getNumIntegers(Cbc_Model *model);

    int Cbc_getNumElements(Cbc_Model *model);

    int Cbc_getRowNz(Cbc_Model *model, int row);

    int *Cbc_getRowIndices(Cbc_Model *model, int row);

    double *Cbc_getRowCoeffs(Cbc_Model *model, int row);

    double Cbc_getRowRHS(Cbc_Model *model, int row);

    char Cbc_getRowSense(Cbc_Model *model, int row);

    int Cbc_getColNz(Cbc_Model *model, int col);

    int *Cbc_getColIndices(Cbc_Model *model, int col);

    double *Cbc_getColCoeffs(Cbc_Model *model, int col);

    void Cbc_addCol(Cbc_Model *model, const char *name,
        double lb, double ub, double obj, char isInteger,
        int nz, int *rows, double *coefs);

    void Cbc_addRow(Cbc_Model *model, const char *name, int nz,
        const int *cols, const double *coefs, char sense, double rhs);

    void Cbc_setObjCoeff(Cbc_Model *model, int index, double value);

    double Cbc_getObjSense(Cbc_Model *model);

    const double *Cbc_getObjCoefficients(Cbc_Model *model);

    const double *Cbc_getColSolution(Cbc_Model *model);

    const double *Cbc_getReducedCost(Cbc_Model *model);

    double *Cbc_bestSolution(Cbc_Model *model);

    int Cbc_numberSavedSolutions(Cbc_Model *model);

    const double *Cbc_savedSolution(Cbc_Model *model, int whichSol);

    double Cbc_savedSolutionObj(Cbc_Model *model, int whichSol);

    double Cbc_getObjValue(Cbc_Model *model);

    void Cbc_setObjSense(Cbc_Model *model, double sense);

    int Cbc_isProvenOptimal(Cbc_Model *model);

    int Cbc_isProvenInfeasible(Cbc_Model *model);

    int Cbc_isContinuousUnbounded(Cbc_Model *model);

    int Cbc_isAbandoned(Cbc_Model *model);

    const double *Cbc_getColLower(Cbc_Model *model);

    const double *Cbc_getColUpper(Cbc_Model *model);

    void Cbc_setColLower(Cbc_Model *model, int index, double value);

    void Cbc_setColUpper(Cbc_Model *model, int index, double value);

    int Cbc_isInteger(Cbc_Model *model, int i);

    void Cbc_getColName(Cbc_Model *model,
        int iColumn, char *name, size_t maxLength);

    void Cbc_getRowName(Cbc_Model *model,
        int iRow, char *name, size_t maxLength);

    void Cbc_setContinuous(Cbc_Model *model, int iColumn);

    void Cbc_setInteger(Cbc_Model *model, int iColumn);

    void Cbc_setParameter(Cbc_Model *model, const char *name,
        const char *value);

    double Cbc_getCutoff(Cbc_Model *model);

    void Cbc_setCutoff(Cbc_Model *model, double cutoff);

    double Cbc_getAllowableGap(Cbc_Model *model);

    void Cbc_setAllowableGap(Cbc_Model *model, double allowedGap);

    double Cbc_getAllowableFractionGap(Cbc_Model *model);

    void Cbc_setAllowableFractionGap(Cbc_Model *model,
        double allowedFracionGap);

    double Cbc_getAllowablePercentageGap(Cbc_Model *model);

    void Cbc_setAllowablePercentageGap(Cbc_Model *model,
        double allowedPercentageGap);

    double Cbc_getMaximumSeconds(Cbc_Model *model);

    void Cbc_setMaximumSeconds(Cbc_Model *model, double maxSeconds);

    int Cbc_getMaximumNodes(Cbc_Model *model);

    void Cbc_setMaximumNodes(Cbc_Model *model, int maxNodes);

    int Cbc_getMaximumSolutions(Cbc_Model *model);

    void Cbc_setMaximumSolutions(Cbc_Model *model, int maxSolutions);

    int Cbc_getLogLevel(Cbc_Model *model);

    void Cbc_setLogLevel(Cbc_Model *model, int logLevel);

    double Cbc_getBestPossibleObjValue(Cbc_Model *model);

    void Cbc_setMIPStart(Cbc_Model *model, int count,
        const char **colNames, const double colValues[]);

    void Cbc_setMIPStartI(Cbc_Model *model, int count, const int colIdxs[],
        const double colValues[]);

    int Cbc_solve(Cbc_Model *model);

    void *Cbc_deleteModel(Cbc_Model *model);

    int Osi_getNumCols( void *osi );

    void Osi_getColName( void *osi, int i, char *name, int maxLen );

    const double *Osi_getColLower( void *osi );

    const double *Osi_getColUpper( void *osi );

    int Osi_isInteger( void *osi, int col );

    int Osi_getNumRows( void *osi );

    int Osi_getRowNz(void *osi, int row);

    const int *Osi_getRowIndices(void *osi, int row);

    const double *Osi_getRowCoeffs(void *osi, int row);

    double Osi_getRowRHS(void *osi, int row);

    char Osi_getRowSense(void *osi, int row);

    const double *Osi_getColSolution(void *osi);

    void OsiCuts_addRowCut( void *osiCuts, int nz, const int *idx,
        const double *coef, char sense, double rhs );

    void Cbc_deleteRows(Cbc_Model *model, int numRows, const int rows[]);

    void Cbc_deleteCols(Cbc_Model *model, int numCols, const int cols[]);

    void Cbc_storeNameIndexes(Cbc_Model *model, char _store);

    int Cbc_getColNameIndex(Cbc_Model *model, const char *name);

    int Cbc_getRowNameIndex(Cbc_Model *model, const char *name);

    void Cbc_addCutCallback(
        void *model, cbc_cut_callback cutcb,
        const char *name, void *appData );

    void Cbc_addIncCallback(
        void *model, cbc_incumbent_callback inccb,
        void *appData );
    """)

CHAR_ONE = "{}".format(chr(1)).encode("utf-8")
CHAR_ZERO = "\0".encode("utf-8")


def cbc_set_parameter(model: Model, param: str, value: str):
    cbclib.Cbc_setParameter(model._model, param.encode("utf-8"),
                            value.encode("utf-8"))


class SolverCbc(Solver):
    def __init__(self, model: Model, name: str, sense: str):
        super().__init__(model, name, sense)

        self._model = cbclib.Cbc_newModel()
        cbclib.Cbc_storeNameIndexes(self._model, CHAR_ONE)

        self._objconst = 0.0

        # to not add cut generators twice when reoptimizing
        self.added_cut_callback = False
        self.added_inc_callback = False

        # setting objective sense
        if sense == MAXIMIZE:
            cbclib.Cbc_setObjSense(self._model, -1.0)

        self.emphasis = SearchEmphasis.DEFAULT
        self.__threads = 0
        self.__verbose = 1
        # pre-allocate temporary space to query names
        self.__name_space = ffi.new("char[{}]".format(MAX_NAME_SIZE))
        # in cut generation
        self.__name_spacec = ffi.new("char[{}]".format(MAX_NAME_SIZE))

    def add_var(self,
                obj: float = 0,
                lb: float = 0,
                ub: float = float("inf"),
                coltype: str = "C",
                column: "Column" = None,
                name: str = ""):
        # collecting column data
        numnz = 0 if column is None else len(column.constrs)
        if not numnz:
            vind = ffi.NULL
            vval = ffi.NULL
        else:
            vind = ffi.new("int[]", [c.idx for c in column.constrs])
            vval = ffi.new("double[]", [coef for coef in column.coeffs])

        isInt = \
            CHAR_ONE if coltype.upper() == "B" or coltype.upper() == "I" \
            else CHAR_ZERO
        cbclib.Cbc_addCol(
            self._model, name.encode("utf-8"),
            lb, ub, obj,
            isInt, numnz, vind, vval)

    def get_objective_const(self) -> float:
        return self._objconst

    def set_objective(self, lin_expr: "LinExpr", sense: str = "") -> None:
        # collecting variable coefficients

        for var, coeff in lin_expr.expr.items():
            cbclib.Cbc_setObjCoeff(self._model, var.idx, coeff)

        # objective function constant
        self._objconst = lin_expr.const

        # setting objective sense
        if sense == MAXIMIZE:
            cbclib.Cbc_setObjSense(self._model, -1.0)
        elif sense == MINIMIZE:
            cbclib.Cbc_setObjSense(self._model, 1.0)

    def relax(self):
        for var in self.model.vars:
            if cbclib.Cbc_isInteger(self._model, var.idx):
                cbclib.Cbc_setContinuous(self._model, var.idx)

    def get_max_seconds(self) -> float:
        return cbclib.Cbc_getMaximumSeconds(self._model)

    def set_max_seconds(self, max_seconds: float):
        cbclib.Cbc_setMaximumSeconds(self._model, max_seconds)

    def get_max_solutions(self) -> int:
        return cbclib.Cbc_getMaximumSolutions(self._model)

    def set_max_solutions(self, max_solutions: int):
        cbclib.Cbc_setMaximumSolutions(self._model, max_solutions)

    def get_max_nodes(self) -> int:
        return cbclib.Cbc_getMaximumNodes(self._model)

    def set_max_nodes(self, max_nodes: int):
        cbclib.Cbc_setMaximumNodes(self._model, max_nodes)

    def get_verbose(self) -> int:
        return self.__verbose

    def set_verbose(self, verbose: int):
        self.__verbose = verbose

    def var_set_var_type(self, var: "Var", value: str):
        cv = var.var_type
        if (value == cv):
            return
        if cv == CONTINUOUS:
            if value == INTEGER or value == BINARY:
                cbclib.Cbc_setInteger(self._model, var.idx)
        else:
            if value == CONTINUOUS:
                cbclib.Cbc_setContinuous(self._model, var.idx)
        if value == BINARY:
            # checking bounds
            if var.lb != 0.0:
                var.lb = 0.0
            if var.ub != 1.0:
                var.ub = 1.0

    def optimize(self) -> OptimizationStatus:

        # get name indexes from an osi problem
        def cbc_get_osi_name_indexes(osi_solver: CData) -> Dict[str, int]:
            nameIdx = {}
            n = cbclib.Osi_getNumCols(osi_solver)
            for i in range(n):
                cbclib.Osi_getColName(osi_solver, i, self.__name_spacec,
                                      MAX_NAME_SIZE)
                cname = ffi.string(self.__name_spacec).decode('utf-8')
                nameIdx[cname] = i

            return nameIdx

        # incumbent callback
        def cbc_inc_callback(cbc_model: CData,
                             obj: float, nz: int,
                             colNames: CData,
                             colValues: CData,
                             appData: CData):
            return

        # cut callback
        @ffi.callback("""
            void (void *osi_solver, void *osi_cuts, void *app_data)
        """)
        def cbc_cut_callback(osi_solver: CData, osi_cuts: CData,
                             app_data: CData):
            global warningMessages
            if osi_solver == ffi.NULL or osi_cuts == ffi.NULL:
                return
            # getting fractional solution
            fracSol = []
            n = cbclib.Osi_getNumCols(osi_solver)
            namespc = ffi.new("char[{}]".format(MAX_NAME_SIZE))
            x = cbclib.Osi_getColSolution(osi_solver)
            if x == ffi.NULL:
                raise Exception('No fractional solution available in callback')

            nnz = 0
            for i in range(n):
                val = float(x[i])
                if abs(val) < 1e-7:
                    continue

                nnz += 1
                cbclib.Osi_getColName(osi_solver, i, namespc, MAX_NAME_SIZE)
                cname = ffi.string(namespc).decode('utf-8')
                var = self.model.var_by_name(cname)
                if var is None and warningMessages == 1:
                    print('-->> var {} not found'.format(cname))
                fracSol.append((var, val))

            if nnz == 0:
                return

            # calling cut generators
            if self.model.cuts_generator is not None:
                cuts = self.model.cuts_generator.generate_cuts(fracSol)

                if len(cuts) == 0:
                    return

                name_idx = cbc_get_osi_name_indexes(osi_solver)

                # translating cuts for variables in the preprocessed problem
                for cut in cuts:
                    if len(cut.expr) == 0:
                        continue
                    cut_idx = []
                    cut_coef = []
                    has_vars = True  # vars not erased in pre-proc
                    missing_var = ''
                    for v, c in cut.expr.items():
                        if v.name in name_idx.keys():
                            cut_idx.append(name_idx[v.name])
                            cut_coef.append(c)
                        else:
                            has_vars = False
                            missing_var = v.name
                            break
                    if has_vars:
                        nz = len(cut_idx)
                        cidx = ffi.new("int[]", cut_idx)
                        cval = ffi.new("double[]", cut_coef)
                        sense = cut.sense.encode('utf-8')
                        rhs = -cut.const
                        cbclib.OsiCuts_addRowCut(
                            osi_cuts, nz, cidx, cval, sense, rhs)
                    else:
                        if warningMessages < 5:
                            print('cut discarded because variable {} does not \
                            exists in preprocessed problem.'.format(
                                missing_var))
                            warningMessages += 1

        # adding cut generators
        m = self.model
        if m.cuts_generator is not None and self.added_cut_callback is False:
            cbclib.Cbc_addCutCallback(self._model, cbc_cut_callback,
                                      'mipCutGen'.encode('utf-8'), ffi.NULL)
            self.added_cut_callback = True

        if self.__verbose == 0:
            cbc_set_parameter(self, 'log', '0')
        else:
            cbc_set_parameter(self, 'log', '1')

        if self.emphasis == SearchEmphasis.FEASIBILITY:
            cbc_set_parameter(self, 'passf', '50')
            cbc_set_parameter(self, 'proximity', 'on')
        if self.emphasis == SearchEmphasis.OPTIMALITY:
            cbc_set_parameter(self, 'strong', '10')
            cbc_set_parameter(self, 'trust', '20')
            cbc_set_parameter(self, 'lagomory', 'endonly')
            cbc_set_parameter(self, 'latwomir', 'endonly')

        if self.model.cuts == 0:
            cbc_set_parameter(self, 'cuts', 'off')

        if self.model.cuts >= 1:
            cbc_set_parameter(self, 'cuts', 'on')
        if self.model.cuts >= 2:
            cbc_set_parameter(self, 'lagomory',
                              'endcleanroot')
            cbc_set_parameter(self, 'latwomir',
                              'endcleanroot')
            cbc_set_parameter(self, 'passC', '-25')
        if self.model.cuts >= 3:
            cbc_set_parameter(self, 'passC', '-35')
            cbc_set_parameter(self, 'lift', 'ifmove')

        if (self.__threads >= 1):
            cbc_set_parameter(self, 'threads',
                              '{}'.format(self.__threads))
        elif self.__threads == -1:
            import multiprocessing
            cbc_set_parameter(self, 'threads',
                              '{}'.format(multiprocessing.cpu_count()))

        cbc_set_parameter(self, 'maxSavedSolutions', '10')
        cbclib.Cbc_solve(self._model)

        if cbclib.Cbc_isAbandoned(self._model):
            return OptimizationStatus.ERROR

        if cbclib.Cbc_isProvenOptimal(self._model):
            return OptimizationStatus.OPTIMAL

        if cbclib.Cbc_isProvenInfeasible(self._model):
            return OptimizationStatus.INFEASIBLE

        if cbclib.Cbc_isContinuousUnbounded(self._model):
            return OptimizationStatus.UNBOUNDED

        if cbclib.Cbc_getNumIntegers(self._model):
            if cbclib.Cbc_bestSolution(self._model):
                return OptimizationStatus.FEASIBLE

        return OptimizationStatus.INFEASIBLE

    def get_objective_sense(self) -> str:
        obj = cbclib.Cbc_getObjSense(self._model)
        if obj < 0.0:
            return MAXIMIZE

        return MINIMIZE

    def set_objective_sense(self, sense: str):
        if sense.strip().upper() == MAXIMIZE.strip().upper():
            cbclib.Cbc_setObjSense(self._model, -1.0)
        elif sense.strip().upper() == MINIMIZE.strip().upper():
            cbclib.Cbc_setObjSense(self._model, 1.0)
        else:
            raise Exception("Unknown sense: {}, use {} or {}".format(sense,
                                                                     MAXIMIZE,
                                                                     MINIMIZE))

    def get_objective_value(self) -> float:
        return cbclib.Cbc_getObjValue(self._model)

    def get_objective_bound(self) -> float:
        return cbclib.Cbc_getBestPossibleObjValue(self._model)

    def var_get_x(self, var: Var) -> float:
        if cbclib.Cbc_getNumIntegers(self._model) > 0:
            x = cbclib.Cbc_bestSolution(self._model)
        else:
            x = cbclib.Cbc_getColSolution(self._model)
        if x == ffi.NULL:
            raise Exception('no solution found')
        return float(x[var.idx])

    def get_num_solutions(self) -> int:
        return cbclib.Cbc_numberSavedSolutions(self._model)

    def get_objective_value_i(self, i: int) -> float:
        return cbclib.Cbc_savedSolutionObj(self._model, i)

    def var_get_xi(self, var: "Var", i: int) -> float:
        x = cbclib.Cbc_savedSolution(self._model, i)
        if x == ffi.NULL:
            raise Exception('no solution available')
        return float(x[var.idx])

    def var_get_rc(self, var: Var) -> float:
        rc = cbclib.Cbc_getReducedCost(self._model)
        if rc == ffi.NULL:
            raise Exception('reduced cost not available')
        return float(rc[var.idx])

    def var_get_lb(self, var: "Var") -> float:
        lb = cbclib.Cbc_getColLower(self._model)
        if lb == ffi.NULL:
            raise Exception('Error while getting lower bound of variables')
        return float(lb[var.idx])

    def var_get_ub(self, var: "Var") -> float:
        ub = cbclib.Cbc_getColLower(self._model)
        if ub == ffi.NULL:
            raise Exception('Error while getting upper bound of variables')
        return float(ub[var.idx])

    def var_get_name(self, idx: int) -> str:
        namep = self.__name_space
        cbclib.Cbc_getColName(self._model, idx, namep, MAX_NAME_SIZE)
        return ffi.string(namep).decode('utf-8')

    def var_get_index(self, name: str) -> int:
        return cbclib.Cbc_getColNameIndex(self._model, name.encode("utf-8"))

    def constr_get_index(self, name: str) -> int:
        return cbclib.Cbc_getRowNameIndex(name.encode("utf-8"))

    def var_get_obj(self, var: Var) -> float:
        obj = cbclib.Cbc_getObjCoefficients(self._model)
        if obj == ffi.NULL:
            raise Exception("Error getting objective function coefficients")
        return obj[var.idx]

    def var_get_var_type(self, var: "Var") -> str:
        isInt = cbclib.Cbc_isInteger(self._model, var.idx)
        if isInt:
            lb = self.var_get_lb(var)
            ub = self.var_get_ub(var)
            if abs(lb) <= 1e-15 and abs(ub - 1.0) <= 1e-15:
                return BINARY
            else:
                return INTEGER

        return CONTINUOUS

    def var_get_column(self, var: "Var") -> Column:
        numnz = cbclib.Cbc_getColNz(self._model, var.idx)

        cidx = cbclib.Cbc_getColIndices(self._model, var.idx)
        if cidx == ffi.NULL:
            raise Exception("Error getting column indices'")
        ccoef = cbclib.Cbc_getColCoeffs(self._model, var.idx)

        col = Column()

        for i in range(numnz):
            col.constrs.append(Constr(self, cidx[i]))
            col.coeffs.append(ccoef[i])

        return col

    def add_constr(self, lin_expr: LinExpr, name: str = ""):
        # collecting linear expression data
        numnz = len(lin_expr.expr)

        cind = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
        cval = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])

        # constraint sense and rhs
        sense = lin_expr.sense.encode("utf-8")
        rhs = -lin_expr.const

        namestr = name.encode("utf-8")
        mp = self._model
        cbclib.Cbc_addRow(mp, namestr, numnz, cind, cval, sense, rhs)

    def write(self, file_path: str):
        fpstr = file_path.encode("utf-8")
        if ".mps" in file_path.lower():
            cbclib.Cbc_writeMps(self._model, fpstr)
        elif ".lp" in file_path.lower():
            cbclib.Cbc_writeLp(self._model, fpstr)
        else:
            raise Exception("Enter a valid extension (.lp or .mps) \
                to indicate the file format")

    def read(self, file_path: str) -> None:
        if not isfile(file_path):
            raise Exception('File {} does not exists'.format(file_path))

        fpstr = file_path.encode("utf-8")
        if ".mps" in file_path.lower():
            cbclib.Cbc_readMps(self._model, fpstr)
        elif ".lp" in file_path.lower():
            cbclib.Cbc_readLp(self._model, fpstr)
        else:
            raise Exception("Enter a valid extension (.lp or .mps) \
                to indicate the file format")

    def set_start(self, start: List[Tuple[Var, float]]) -> None:
        n = len(start)
        dv = ffi.new("double[]", [start[i][1] for i in range(n)])
        iv = ffi.new("int[]", [start[i][0].idx for i in range(n)])
        mdl = self._model
        cbclib.Cbc_setMIPStartI(mdl, n, iv, dv)

    def num_cols(self) -> int:
        return cbclib.Cbc_getNumCols(self._model)

    def num_int(self) -> int:
        return cbclib.Cbc_getNumIntegers(self._model)

    def num_rows(self) -> int:
        return cbclib.Cbc_getNumRows(self._model)

    def num_nz(self) -> int:
        return cbclib.Cbc_getNumElements(self._model)

    def get_cutoff(self) -> float:
        return cbclib.Cbc_getCutoff(self._model)

    def set_cutoff(self, cutoff: float):
        cbclib.Cbc_setCutoff(self._model, cutoff)

    def get_mip_gap_abs(self) -> float:
        return cbclib.Cbc_getAllowableGap(self._model)

    def set_mip_gap_abs(self, allowable_gap: float):
        cbclib.Cbc_setAllowableGap(self._model, allowable_gap)

    def get_mip_gap(self) -> float:
        return cbclib.Cbc_getAllowableFractionGap(self._model)

    def set_mip_gap(self, allowable_ratio_gap: float):
        cbclib.Cbc_setAllowableFractionGap(self._model, allowable_ratio_gap)

    def constr_get_expr(self, constr: Constr) -> LinExpr:
        numnz = cbclib.Cbc_getRowNz(self._model, constr.idx)

        ridx = cbclib.Cbc_getRowIndices(self._model, constr.idx)
        if ridx == ffi.NULL:
            raise Exception("Error getting row indices.")
        rcoef = cbclib.Cbc_getRowCoeffs(self._model, constr.idx)
        if rcoef == ffi.NULL:
            raise Exception("Error getting row coefficients.")

        rhs = cbclib.Cbc_getRowRHS(self._model, constr.idx)
        rsense = cbclib.Cbc_getRowSense(self._model,
                                        constr.idx).decode("utf-8").upper()
        sense = ''
        if (rsense == 'E'):
            sense = EQUAL
        elif (rsense == 'L'):
            sense = LESS_OR_EQUAL
        elif (rsense == 'G'):
            sense = GREATER_OR_EQUAL
        else:
            raise Exception('Unknow sense: {}'.format(rsense))

        expr = LinExpr(const=-rhs, sense=sense)
        for i in range(numnz):
            expr.add_var(self.model.vars[ridx[i]], rcoef[i])

        return expr

    def constr_get_name(self, idx: int) -> str:
        cbclib.Cbc_getRowName(self._model, idx,
                              self.__name_space, MAX_NAME_SIZE)
        return self.__name_space.decode('utf-8')

    def set_processing_limits(self,
                              maxTime=INF,
                              maxNodes=INF,
                              maxSol=INF):
        if maxTime != INF:
            cbc_set_parameter(self, 'timeMode', 'elapsed')
            cbc_set_parameter(self, 'seconds', '{}'.format(maxTime))
        if maxNodes != INF:
            cbc_set_parameter(self, 'maxNodes', '{}'.format(maxNodes))
        if maxSol != INF:
            cbc_set_parameter(self, 'maxSolutions', '{}'.format(maxSol))

    def get_emphasis(self) -> SearchEmphasis:
        return self.emphasis

    def set_emphasis(self, emph: SearchEmphasis):
        self.emphasis = emph

    def set_num_threads(self, threads: int):
        self.__threads = threads

    def remove_constrs(self, constrs: List[int]):
        idx = ffi.new("int[]", constrs)
        cbclib.Cbc_deleteRows(self._model, idx)

    def remove_vars(self, cols: List[int]):
        idx = ffi.new("int[]", cols)
        cbclib.Cbc_deleteCols(self._model, idx)

    def __del__(self):
        cbclib.Cbc_deleteModel(self._model)


# vim: ts=4 sw=4 et
