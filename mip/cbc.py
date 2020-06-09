"""Python-MIP interface to the COIN-OR Branch-and-Cut solver CBC"""

import logging
from typing import Dict, List, Tuple, Optional, Union
from sys import platform, maxsize
from os.path import dirname, isfile
import os
import multiprocessing as multip
import numbers
from cffi import FFI
from mip.model import xsum
import mip
from mip.lists import EmptyVarSol, EmptyRowSol
from mip.exceptions import (
    ParameterNotAvailable,
    InvalidParameter,
    MipBaseException,
)
from mip import (
    Model,
    Var,
    Constr,
    Column,
    LinExpr,
    VConstrList,
    VVarList,
    Solver,
    MAXIMIZE,
    SearchEmphasis,
    CONTINUOUS,
    BINARY,
    INTEGER,
    MINIMIZE,
    EQUAL,
    LESS_OR_EQUAL,
    GREATER_OR_EQUAL,
    OptimizationStatus,
    LP_Method,
    CutType,
    CutPool,
)

logger = logging.getLogger(__name__)
warningMessages = 0

ffi = FFI()
has_cbc = False
os_is_64_bit = maxsize > 2 ** 32
INF = float("inf")
cut_idx = 0

# for variables and rows
MAX_NAME_SIZE = 512

DEF_PUMPP = 30

try:
    pathmip = dirname(mip.__file__)
    pathlib = os.path.join(pathmip, "libraries")
    libfile = ""
    # if user wants to force the loading of an specific CBC library
    # (for debugging purposes, for example)
    if "PMIP_CBC_LIBRARY" in os.environ:
        libfile = os.environ["PMIP_CBC_LIBRARY"]
        pathlib = dirname(libfile)

        if platform.lower().startswith("win"):
            if pathlib not in os.environ["PATH"]:
                os.environ["PATH"] += ";" + pathlib
    else:
        if "linux" in platform.lower():
            if os_is_64_bit:
                pathlib = os.path.join(pathlib, "lin64")
                libfile = os.path.join(pathlib, "libCbcSolver.so")
            else:
                raise NotImplementedError("Linux 32 bits platform not supported.")
        elif platform.lower().startswith("win"):
            if os_is_64_bit:
                pathlib = os.path.join(pathlib, "win64")
                if pathlib not in os.environ["PATH"]:
                    os.environ["PATH"] = pathlib + ";" + os.environ["PATH"]
                libfile = os.path.join(pathlib, "libCbcSolver-0.dll")
            else:
                raise NotImplementedError("Win32 platform not supported.")
        elif platform.lower().startswith("darwin") or platform.lower().startswith(
            "macos"
        ):
            if os_is_64_bit:
                libfile = os.path.join(pathlib, "cbc-c-darwin-x86-64.dylib")
        if not libfile:
            raise NotImplementedError("You operating system/platform is not supported")
    old_dir = os.getcwd()
    os.chdir(pathlib)
    cbclib = ffi.dlopen(libfile)
    os.chdir(old_dir)
    has_cbc = True
except Exception as e:
    logger.error("An error occurred while loading the CBC library:\t " "{}\n".format(e))
    has_cbc = False

if has_cbc:
    ffi.cdef(
        """
    typedef int(*cbc_progress_callback)(void *model,
                                        int phase,
                                        int step,
                                        const char *phaseName,
                                        double seconds,
                                        double lb,
                                        double ub,
                                        int nint,
                                        int *vecint,
                                        void *cbData
                                        );

    typedef void(*cbc_callback)(void *model, int msgno, int ndouble,
        const double *dvec, int nint, const int *ivec,
        int nchar, char **cvec);

    typedef void(*cbc_cut_callback)(void *osiSolver,
        void *osiCuts, void *appdata);

    typedef int (*cbc_incumbent_callback)(void *cbcModel,
        double obj, int nz,
        char **vnames, double *x, void *appData);

    typedef void Cbc_Model;

    void *Cbc_newModel();

    void Cbc_readLp(Cbc_Model *model, const char *file);

    void Cbc_readMps(Cbc_Model *model, const char *file);

    char Cbc_supportsGzip();

    char Cbc_supportsBzip2();

    void Cbc_writeLp(Cbc_Model *model, const char *file);

    void Cbc_writeMps(Cbc_Model *model, const char *file);

    int Cbc_getNumCols(Cbc_Model *model);

    int Cbc_getNumRows(Cbc_Model *model);

    int Cbc_getNumIntegers(Cbc_Model *model);

    int Cbc_getNumElements(Cbc_Model *model);

    int Cbc_getRowNz(Cbc_Model *model, int row);

    int *Cbc_getRowIndices(Cbc_Model *model, int row);

    double *Cbc_getRowCoeffs(Cbc_Model *model, int row);

    double Cbc_getRowRHS(Cbc_Model *model, int row);

    void Cbc_setRowRHS(Cbc_Model *model, int row, double rhs);

    char Cbc_getRowSense(Cbc_Model *model, int row);

    const double *Cbc_getRowActivity(Cbc_Model *model);

    const double *Cbc_getRowSlack(Cbc_Model *model);

    int Cbc_getColNz(Cbc_Model *model, int col);

    int *Cbc_getColIndices(Cbc_Model *model, int col);

    double *Cbc_getColCoeffs(Cbc_Model *model, int col);

    void Cbc_addCol(Cbc_Model *model, const char *name,
        double lb, double ub, double obj, char isInteger,
        int nz, int *rows, double *coefs);

    void Cbc_addRow(Cbc_Model *model, const char *name, int nz,
        const int *cols, const double *coefs, char sense, double rhs);

    void Cbc_addLazyConstraint(Cbc_Model *model, int nz,
        int *cols, double *coefs, char sense, double rhs);

    void Cbc_addSOS(Cbc_Model *model, int numRows, const int *rowStarts,
        const int *colIndices, const double *weights, const int type);

    int Cbc_numberSOS(Cbc_Model *model);

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

    double Cbc_getColObj(Cbc_Model *model, int colIdx);

    double Cbc_getColLB(Cbc_Model *model, int colIdx);

    double Cbc_getColUB(Cbc_Model *model, int colIdx);

    void Cbc_setColLower(Cbc_Model *model, int index, double value);

    void Cbc_setColUpper(Cbc_Model *model, int index, double value);

    int Cbc_isInteger(Cbc_Model *model, int i);

    void Cbc_getColName(Cbc_Model *model,
        int iColumn, char *name, size_t maxLength);

    void Cbc_getRowName(Cbc_Model *model,
        int iRow, char *name, size_t maxLength);

    void Cbc_setContinuous(Cbc_Model *model, int iColumn);

    void Cbc_setInteger(Cbc_Model *model, int iColumn);

    /*! Integer parameters */
    enum IntParam {
    INT_PARAM_PERT_VALUE          = 0,  /*! Method of perturbation, -5000 to 102, default 50 */
    INT_PARAM_IDIOT               = 1,  /*! Parameter of the "idiot" method to try to produce an initial feasible basis. -1 let the solver decide if this should be applied; 0 deactivates it and >0 sets number of passes. */
    INT_PARAM_STRONG_BRANCHING    = 2,  /*! Number of variables to be evaluated in strong branching. */
    INT_PARAM_CUT_DEPTH           = 3,  /*! Sets the application of cuts to every depth multiple of this value. -1, the default value, let the solve decide. */
    INT_PARAM_MAX_NODES           = 4,  /*! Maximum number of nodes to be explored in the search tree */
    INT_PARAM_NUMBER_BEFORE       = 5,  /*! Number of branche before trusting pseudocodes computed in strong branching. */
    INT_PARAM_FPUMP_ITS           = 6,  /*! Maximum number of iterations in the feasibility pump method. */
    INT_PARAM_MAX_SOLS            = 7,  /*! Maximum number of solutions generated during the search. Stops the search when this number of solutions is found. */
    INT_PARAM_CUT_PASS_IN_TREE    = 8, /*! Maxinum number of cuts passes in the search tree (with the exception of the root node). Default 1. */
    INT_PARAM_THREADS             = 9, /*! Number of threads that can be used in the branch-and-bound method.*/
    INT_PARAM_CUT_PASS            = 10, /*! Number of cut passes in the root node. Default -1, solver decides */
    INT_PARAM_LOG_LEVEL           = 11, /*! Verbosity level, from 0 to 2 */
    INT_PARAM_MAX_SAVED_SOLS      = 12, /*! Size of the pool to save the best solutions found during the search. */
    INT_PARAM_MULTIPLE_ROOTS      = 13, /*! Multiple root passes to get additional cuts and solutions. */
    INT_PARAM_ROUND_INT_VARS      = 14, /*! If integer variables should be round to remove small infeasibilities. This can increase the overall amount of infeasibilities in problems with both continuous and integer variables */
    INT_PARAM_RANDOM_SEED         = 15, /*! When solving LP and MIP, randomization is used to break ties in some decisions. This changes the random seed so that multiple executions can produce different results */
    INT_PARAM_ELAPSED_TIME        = 16  /*! When =1 use elapsed (wallclock) time, otherwise use CPU time */
    };
#define N_INT_PARAMS 17
    void Cbc_setIntParam(Cbc_Model *model, enum IntParam which, const int val);

enum DblParam {
  DBL_PARAM_PRIMAL_TOL    = 0,  /*! Tollerance to consider a solution feasible in the linear programming solver. */
  DBL_PARAM_DUAL_TOL      = 1,  /*! Tollerance for a solution to be considered optimal in the linear programming solver. */
  DBL_PARAM_ZERO_TOL      = 2,  /*! Coefficients less that this value will be ignored when reading instances */
  DBL_PARAM_INT_TOL       = 3,  /*! Maximum allowed distance from integer value for a variable to be considered integral */
  DBL_PARAM_PRESOLVE_TOL  = 4,  /*! Tollerance used in the presolver, should be increased if the pre-solver is declaring infeasible a feasible problem */
  DBL_PARAM_TIME_LIMIT    = 5,  /*! Time limit in seconds */
  DBL_PARAM_PSI           = 6,  /*! Two dimensional princing factor in the Positive Edge pivot strategy. */
  DBL_PARAM_CUTOFF        = 7,  /*! Only search for solutions with cost less-or-equal to this value. */
  DBL_PARAM_ALLOWABLE_GAP = 8,  /*! Allowable gap between the lower and upper bound to conclude the search */
  DBL_PARAM_GAP_RATIO     = 9   /*! Stops the search when the difference between the upper and lower bound is less than this fraction of the larger value */
};
#define N_DBL_PARAMS 10

    void Cbc_setDblParam(Cbc_Model *model, enum DblParam which, const double val);


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

    enum LPMethod {
      LPM_Auto    = 0,  /*! Solver will decide automatically which method to use */
      LPM_Dual    = 1,  /*! Dual simplex */
      LPM_Primal  = 2,  /*! Primal simplex */
      LPM_Barrier = 3   /*! The barrier algorithm. */
    };

    void
    Cbc_setLPmethod(Cbc_Model *model, enum LPMethod lpm );

    void Cbc_updateConflictGraph( Cbc_Model *model );

    const void *Cbc_conflictGraph( Cbc_Model *model );

    int Cbc_solve(Cbc_Model *model);

    int Cbc_solveLinearProgram(Cbc_Model *model);

    enum CutType {
      CT_Gomory         = 0,  /*! Gomory cuts obtained from the tableau */
      CT_MIR            = 1,  /*! Mixed integer rounding cuts */
      CT_ZeroHalf       = 2,  /*! Zero-half cuts */
      CT_Clique         = 3,  /*! Clique cuts */
      CT_KnapsackCover  = 4,  /*! Knapsack cover cuts */
      CT_LiftAndProject = 5   /*! Lift and project cuts */
    };

    void Cgl_generateCuts( void *osiClpSolver, enum CutType ct, void *osiCuts, int strength );

    void *Cbc_getSolverPtr(Cbc_Model *model);

    void *Cbc_deleteModel(Cbc_Model *model);

    int Osi_getNumIntegers( void *osi );

    const double *Osi_getReducedCost( void *osi );

    const double *Osi_getObjCoefficients();

    double Osi_getObjSense();

    void *Osi_newSolver();

    void Osi_deleteSolver( void *osi );

    void Osi_initialSolve(void *osi);

    void Osi_resolve(void *osi);

    void Osi_branchAndBound(void *osi);

    char Osi_isAbandoned(void *osi);

    char Osi_isProvenOptimal(void *osi);

    char Osi_isProvenPrimalInfeasible(void *osi);

    char Osi_isProvenDualInfeasible(void *osi);

    char Osi_isPrimalObjectiveLimitReached(void *osi);

    char Osi_isDualObjectiveLimitReached(void *osi);

    char Osi_isIterationLimitReached(void *osi);

    double Osi_getObjValue( void *osi );

    void Osi_setColUpper (void *osi, int elementIndex, double ub);

    void Osi_setColLower(void *osi, int elementIndex, double lb);

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

    void Osi_setObjCoef(void *osi, int index, double obj);

    void Osi_setObjSense(void *osi, double sense);

    const double *Osi_getColSolution(void *osi);

    void *OsiCuts_new();

    void OsiCuts_addRowCut( void *osiCuts, int nz, const int *idx,
        const double *coef, char sense, double rhs );

    void OsiCuts_addGlobalRowCut( void *osiCuts, int nz, const int *idx,
        const double *coef, char sense, double rhs );

    int OsiCuts_sizeRowCuts( void *osiCuts );

    int OsiCuts_nzRowCut( void *osiCuts, int iRowCut );

    const int * OsiCuts_idxRowCut( void *osiCuts, int iRowCut );

    const double *OsiCuts_coefRowCut( void *osiCuts, int iRowCut );

    double OsiCuts_rhsRowCut( void *osiCuts, int iRowCut );

    char OsiCuts_senseRowCut( void *osiCuts, int iRowCut );

    void OsiCuts_delete( void *osiCuts );

    void Osi_addCol(void *osi, const char *name, double lb, double ub,
       double obj, char isInteger, int nz, int *rows, double *coefs);

    void Osi_addRow(void *osi, const char *name, int nz,
        const int *cols, const double *coefs, char sense, double rhs);

    void Cbc_deleteRows(Cbc_Model *model, int numRows, const int rows[]);

    void Cbc_deleteCols(Cbc_Model *model, int numCols, const int cols[]);

    void Cbc_storeNameIndexes(Cbc_Model *model, char _store);

    int Cbc_getColNameIndex(Cbc_Model *model, const char *name);

    int Cbc_getRowNameIndex(Cbc_Model *model, const char *name);

    void Cbc_problemName(Cbc_Model *model, int maxNumberCharacters,
                         char *array);

    int Cbc_setProblemName(Cbc_Model *model, const char *array);

    double Cbc_getPrimalTolerance(Cbc_Model *model);

    void Cbc_setPrimalTolerance(Cbc_Model *model, double tol);

    double Cbc_getDualTolerance(Cbc_Model *model);

    void Cbc_setDualTolerance(Cbc_Model *model, double tol);

    void Cbc_addCutCallback(Cbc_Model *model, cbc_cut_callback cutcb,
        const char *name, void *appData, int howOften, char atSolution );

    void Cbc_addIncCallback(
        void *model, cbc_incumbent_callback inccb,
        void *appData );

    void Cbc_registerCallBack(Cbc_Model *model,
        cbc_callback userCallBack);

    void Cbc_addProgrCallback(void *model,
        cbc_progress_callback prgcbc, void *appData);

    void Cbc_clearCallBack(Cbc_Model *model);

    const double *Cbc_getRowPrice(Cbc_Model *model);

    const double *Osi_getRowPrice(void *osi);

    double Osi_getIntegerTolerance(void *osi);

    void Osi_checkCGraph( void *osi );

    const void * Osi_CGraph( void *osi );

    size_t CG_nodes( void *cgraph );

    char CG_conflicting( void *cgraph, int n1, int n2 );

    double CG_density( void *cgraph );

    typedef struct {
      size_t n;
      const size_t *neigh;
    } CGNeighbors;

    CGNeighbors CG_conflictingNodes(Cbc_Model *model, void *cgraph, size_t node);
    """
    )

CHAR_ONE = "{}".format(chr(1)).encode("utf-8")
CHAR_ZERO = "\0".encode("utf-8")

DBL_PARAM_PRIMAL_TOL = 0
DBL_PARAM_DUAL_TOL = 1
DBL_PARAM_ZERO_TOL = 2
DBL_PARAM_INT_TOL = 3
DBL_PARAM_PRESOLVE_TOL = 4
DBL_PARAM_TIME_LIMIT = 5
DBL_PARAM_PSI = 6
DBL_PARAM_CUTOFF = 7
DBL_PARAM_ALLOWABLE_GAP = 8
DBL_PARAM_GAP_RATIO = 9

INT_PARAM_PERT_VALUE = 0
INT_PARAM_IDIOT = 1
INT_PARAM_STRONG_BRANCHING = 2
INT_PARAM_CUT_DEPTH = 3
INT_PARAM_MAX_NODES = 4
INT_PARAM_NUMBER_BEFORE = 5
INT_PARAM_FPUMP_ITS = 6
INT_PARAM_MAX_SOLS = 7
INT_PARAM_CUT_PASS_IN_TREE = 8
INT_PARAM_THREADS = 9
INT_PARAM_CUT_PASS = 10
INT_PARAM_LOG_LEVEL = 11
INT_PARAM_MAX_SAVED_SOLS = 12
INT_PARAM_MULTIPLE_ROOTS = 13
INT_PARAM_ROUND_INT_VARS = 14
INT_PARAM_RANDOM_SEED = 15


Osi_getNumCols = cbclib.Osi_getNumCols
Osi_getColSolution = cbclib.Osi_getColSolution
Osi_getIntegerTolerance = cbclib.Osi_getIntegerTolerance
Osi_isInteger = cbclib.Osi_isInteger
Osi_isProvenOptimal = cbclib.Osi_isProvenOptimal
Cbc_setIntParam = cbclib.Cbc_setIntParam
Cbc_getSolverPtr = cbclib.Cbc_getSolverPtr

Cgl_generateCuts = cbclib.Cgl_generateCuts
Cbc_solveLinearProgram = cbclib.Cbc_solveLinearProgram

OsiCuts_new = cbclib.OsiCuts_new
OsiCuts_addRowCut = cbclib.OsiCuts_addRowCut
OsiCuts_addGlobalRowCut = cbclib.OsiCuts_addGlobalRowCut
OsiCuts_sizeRowCuts = cbclib.OsiCuts_sizeRowCuts
OsiCuts_nzRowCut = cbclib.OsiCuts_nzRowCut
OsiCuts_idxRowCut = cbclib.OsiCuts_idxRowCut
OsiCuts_coefRowCut = cbclib.OsiCuts_coefRowCut
OsiCuts_rhsRowCut = cbclib.OsiCuts_rhsRowCut
OsiCuts_senseRowCut = cbclib.OsiCuts_senseRowCut
OsiCuts_delete = cbclib.OsiCuts_delete


def cbc_set_parameter(model: Solver, param: str, value: str):
    cbclib.Cbc_setParameter(model._model, param.encode("utf-8"), value.encode("utf-8"))


class SolverCbc(Solver):
    def __init__(self, model: Model, name: str, sense: str):
        super().__init__(model, name, sense)

        self._model = cbclib.Cbc_newModel()
        cbclib.Cbc_storeNameIndexes(self._model, CHAR_ONE)

        self.iidx_space = 4096
        self.iidx = ffi.new("int[%d]" % self.iidx_space)
        self.dvec = ffi.new("double[%d]" % self.iidx_space)

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
        self.__log = (
            []
        )  # type: List[Tuple[numbers.Real, Tuple[numbers.Real, numbers.Real]]]
        self.set_problem_name(name)
        self.__pumpp = DEF_PUMPP

        # where solution will be stored
        self.__x = EmptyVarSol(model)
        self.__rc = EmptyVarSol(model)
        self.__pi = EmptyRowSol(model)
        self.__slack = EmptyRowSol(model)
        self.__obj_val = None
        self.__obj_bound = None
        self.__num_solutions = 0

    def __clear_sol(self: "SolverCbc"):
        self.__x = EmptyVarSol(self.model)
        self.__rc = EmptyVarSol(self.model)
        self.__pi = EmptyRowSol(self.model)
        self.__slack = EmptyRowSol(self.model)
        self.__obj_val = None
        self.__obj_bound = None
        self.__num_solutions = 0

    def add_var(
        self,
        obj: numbers.Real = 0,
        lb: numbers.Real = 0,
        ub: numbers.Real = float("inf"),
        coltype: str = "C",
        column: Optional[Column] = None,
        name: str = "",
    ):
        if column is None:
            vind = ffi.NULL
            vval = ffi.NULL
            numnz = 0
        else:
            numnz = len(column.constrs)
            vind = ffi.new("int[]", [c.idx for c in column.constrs])
            vval = ffi.new("double[]", [coef for coef in column.coeffs])

        isInt = (
            CHAR_ONE if coltype.upper() == "B" or coltype.upper() == "I" else CHAR_ZERO
        )
        cbclib.Cbc_addCol(
            self._model, name.encode("utf-8"), lb, ub, obj, isInt, numnz, vind, vval,
        )

    def update_conflict_graph(self: "SolverCbc"):
        cbclib.Cbc_updateConflictGraph(self._model)

    def cgraph_density(self: "SolverCbc") -> float:
        cg = cbclib.Cbc_conflictGraph(self._model)
        if cg == ffi.NULL:
            return 0.0
        return cbclib.CG_density(cg)

    def conflicting(
        self: "SolverCbc", e1: Union["LinExpr", "Var"], e2: Union["LinExpr", "Var"],
    ) -> bool:
        idx1, idx2 = (None, None)
        if isinstance(e1, Var):
            idx1 = e1.idx
        elif isinstance(e1, LinExpr):
            if len(e1.expr) == 1 and e1.sense == EQUAL:
                v1 = next(iter(e1.expr.keys()))
                if abs(e1.const) <= 1e-15:
                    idx1 = v1.idx + v1.model.num_cols
                elif abs(e1.const + 1.0) <= 1e-15:
                    idx1 = v1.idx
                else:
                    raise InvalidParameter(
                        "LinExpr should contain an "
                        "assignment to a binary variable, "
                        "e.g.: x1 == 1"
                    )
            else:
                raise InvalidParameter(
                    "LinExpr should contain an "
                    "assignment to a binary variable, "
                    "e.g.: x1 == 1"
                )
        else:
            raise TypeError("type {} not supported".format(type(e1)))

        if isinstance(e2, Var):
            idx2 = e2.idx
        elif isinstance(e2, LinExpr):
            if len(e2.expr) == 1 and e2.sense == EQUAL:
                v2 = next(iter(e2.expr.keys()))
                if abs(e2.const) <= 1e-15:
                    idx2 = v2.idx + v2.model.num_cols
                elif abs(e2.const + 1.0) <= 1e-15:
                    idx2 = v2.idx
                else:
                    raise InvalidParameter(
                        "LinExpr should contain an "
                        "assignment to a binary variable, "
                        "e.g.: x1 == 1"
                    )
            else:
                raise InvalidParameter(
                    "LinExpr should contain an "
                    "assignment to a binary variable, "
                    "e.g.: x1 == 1"
                )
        else:
            raise TypeError("type {} not supported".format(type(e2)))

        cg = cbclib.Cbc_conflictGraph(self._model)
        if cg == ffi.NULL:
            return False

        return cbclib.CG_conflicting(cg, idx1, idx2) == CHAR_ONE

    def conflicting_nodes(
        self: "SolverCbc", v1: Union["Var", "LinExpr"]
    ) -> Tuple[List["Var"], List["Var"]]:
        """Returns all assignment conflicting with the assignment in v1 in the
        conflict graph.
        """
        cg = cbclib.Cbc_conflictGraph(self._model)
        if cg == ffi.NULL:
            return (list(), list())

        idx1 = None
        if isinstance(v1, Var):
            idx1 = v1.idx
        elif isinstance(v1, LinExpr):
            if len(v1.expr) == 1 and v1.sense == EQUAL:
                var = next(iter(v1.expr.keys()))
                if abs(v1.const) <= 1e-15:
                    idx1 = var.idx + var.model.num_cols
                elif abs(v1.const + 1.0) <= 1e-15:
                    idx1 = var.idx
                else:
                    raise InvalidParameter(
                        "LinExpr should contain an "
                        "assignment to a binary variable, "
                        "e.g.: x1 == 1"
                    )
            else:
                raise InvalidParameter(
                    "LinExpr should contain an "
                    "assignment to a binary variable, "
                    "e.g.: x1 == 1"
                )
        else:
            raise TypeError("type {} not supported".format(type(v1)))

        cgn = cbclib.CG_conflictingNodes(self._model, cg, idx1)
        n = cgn.n
        neighs = cgn.neigh
        cols = self.model.num_cols
        l1, l0 = list(), list()
        for i in range(n):
            if cgn.neigh[i] < cols:
                l1.append(self.model.vars[neighs[i]])
            else:
                l0.append(self.model.vars[neighs[i] - cols])

        return (l1, l0)

    def get_objective_const(self) -> numbers.Real:
        return self._objconst

    def get_objective(self) -> LinExpr:
        obj = cbclib.Cbc_getObjCoefficients(self._model)
        if obj == ffi.NULL:
            raise ParameterNotAvailable("Error getting objective function coefficients")
        return (
            xsum(
                obj[j] * self.model.vars[j]
                for j in range(self.num_cols())
                if abs(obj[j]) >= 1e-15
            )
            + self._objconst
        )

    def set_objective(self, lin_expr: "LinExpr", sense: str = "") -> None:
        # collecting variable coefficients

        c = ffi.new("double[]", [0.0 for i in range(self.num_cols())])
        for var, coeff in lin_expr.expr.items():
            c[var.idx] = coeff

        for i in range(self.num_cols()):
            cbclib.Cbc_setObjCoeff(self._model, i, c[i])

        # objective function constant
        self._objconst = lin_expr.const

        # setting objective sense
        if MAXIMIZE in (lin_expr.sense, sense):
            cbclib.Cbc_setObjSense(self._model, -1.0)
        elif MINIMIZE in (lin_expr.sense, sense):
            cbclib.Cbc_setObjSense(self._model, 1.0)

    def relax(self):
        for var in self.model.vars:
            if cbclib.Cbc_isInteger(self._model, var.idx):
                cbclib.Cbc_setContinuous(self._model, var.idx)

    def get_max_seconds(self) -> numbers.Real:
        return cbclib.Cbc_getMaximumSeconds(self._model)

    def set_max_seconds(self, max_seconds: numbers.Real):
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
        if value == cv:
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

    def var_set_obj(self, var: "Var", value: numbers.Real):
        cbclib.Cbc_setObjCoeff(self._model, var.idx, value)

    def generate_cuts(
        self,
        cut_types: Optional[List[CutType]] = None,
        max_cuts: int = maxsize,
        min_viol: numbers.Real = 1e-4,
    ) -> CutPool:
        cp = CutPool()
        cbc_model = self._model
        osi_solver = Cbc_getSolverPtr(cbc_model)
        osi_cuts = OsiCuts_new()

        if not cut_types:
            cut_types = [e for e in CutType]
        for cut_type in cut_types:
            if self.__verbose >= 1:
                logger.info(
                    "searching for violated " "{} cuts ... ".format(cut_type.name)
                )
            nc1 = OsiCuts_sizeRowCuts(osi_cuts)
            Cgl_generateCuts(osi_solver, int(cut_type.value), osi_cuts, int(1))
            nc2 = OsiCuts_sizeRowCuts(osi_cuts)
            if self.__verbose >= 1:
                logger.info("{} found.\n".format(nc2 - nc1))
            if OsiCuts_sizeRowCuts(osi_cuts) >= max_cuts:
                break

        for i in range(OsiCuts_sizeRowCuts(osi_cuts)):
            rhs = OsiCuts_rhsRowCut(osi_cuts, i)
            rsense = OsiCuts_senseRowCut(osi_cuts, i).decode("utf-8").upper()

            sense = ""
            if rsense == "E":
                sense = EQUAL
            elif rsense == "L":
                sense = LESS_OR_EQUAL
            elif rsense == "G":
                sense = GREATER_OR_EQUAL
            else:
                raise ValueError("Unknow sense: {}".format(rsense))
            idx = OsiCuts_idxRowCut(osi_cuts, i)
            coef = OsiCuts_coefRowCut(osi_cuts, i)
            nz = OsiCuts_nzRowCut(osi_cuts, i)
            model = self.model
            levars = [model.vars[idx[j]] for j in range(nz)]
            lecoefs = [coef[j] for j in range(nz)]
            cut = LinExpr(levars, lecoefs, -rhs, sense)
            if cut.violation < min_viol:
                continue
            cp.add(cut)

        OsiCuts_delete(osi_cuts)
        return cp

    def optimize(self, relax: bool = False) -> OptimizationStatus:
        # get name indexes from an osi problem
        def cbc_get_osi_name_indexes(osi_solver) -> Dict[str, int]:
            nameIdx = {}
            n = cbclib.Osi_getNumCols(osi_solver)
            for i in range(n):
                cbclib.Osi_getColName(osi_solver, i, self.__name_spacec, MAX_NAME_SIZE)
                cname = ffi.string(self.__name_spacec).decode("utf-8")
                nameIdx[cname] = i

            return nameIdx

        # progress callback
        @ffi.callback(
            """
            int (void *, int, int, const char *, double, double, double,
            int, int *, void *)
        """
        )
        def cbc_progress_callback(
            model,
            phase: int,
            step: int,
            phaseName,
            seconds: numbers.Real,
            lb: numbers.Real,
            ub: numbers.Real,
            nint: int,
            vint,
            cbData,
        ) -> int:
            self.__log.append((seconds, (lb, ub)))
            return -1

        # incumbent callback
        def cbc_inc_callback(
            cbc_model, obj: numbers.Real, nz: int, colNames, colValues, appData
        ):
            return

        # cut callback
        @ffi.callback(
            """
            void (void *osi_solver, void *osi_cuts, void *app_data)
        """
        )
        def cbc_cut_callback(osi_solver, osi_cuts, app_data):
            if (
                osi_solver == ffi.NULL
                or osi_cuts == ffi.NULL
                or (
                    self.model.cuts_generator is None
                    and self.model.lazy_constrs_generator is None
                )
            ):
                return
            if Osi_isProvenOptimal(osi_solver) != CHAR_ONE:
                return

            # checking if solution is fractional or not
            nc = Osi_getNumCols(osi_solver)
            x = Osi_getColSolution(osi_solver)
            itol = Osi_getIntegerTolerance(osi_solver)
            fractional = False
            for j in range(nc):
                if Osi_isInteger(osi_solver, j):
                    if abs(x[j] - round(x[j])) > itol:
                        fractional = True
                        break

            osi_model = ModelOsi(osi_solver)
            osi_model._status = osi_model.solver.get_status()
            osi_model.solver.osi_cutsp = osi_cuts
            osi_model.fractional = fractional
            if fractional and self.model.cuts_generator:
                self.model.cuts_generator.generate_constrs(osi_model)
            if (not fractional) and self.model.lazy_constrs_generator:
                self.model.lazy_constrs_generator.generate_constrs(osi_model)

        if self.__verbose == 0:
            cbclib.Cbc_setLogLevel(self._model, 0)
        else:
            cbclib.Cbc_setLogLevel(self._model, 1)

        if relax:
            self.__clear_sol()
            res = Cbc_solveLinearProgram(self._model)
            if res == 0:
                self.__x = cbclib.Cbc_getColSolution(self._model)
                self.__rc = cbclib.Cbc_getReducedCost(self._model)
                self.__pi = cbclib.Cbc_getRowPrice(self._model)
                self.__slack = cbclib.Cbc_getRowSlack(self._model)
                self.__obj_val = cbclib.Cbc_getObjValue(self._model) + self._objconst
                self.__obj_bound = self.__obj_val
                self.__num_solutions = 1

                return OptimizationStatus.OPTIMAL
            if res == 2:
                return OptimizationStatus.UNBOUNDED
            if res == 3:
                return OptimizationStatus.INFEASIBLE
            return OptimizationStatus.ERROR

        # adding cut generators
        m = self.model
        if m.cuts_generator is not None:
            atSol = CHAR_ZERO
            cbclib.Cbc_addCutCallback(
                self._model,
                cbc_cut_callback,
                "UserCuts".encode("utf-8"),
                ffi.NULL,
                1,
                atSol,
            )
        if m.lazy_constrs_generator is not None:
            atSol = CHAR_ONE
            cbc_set_parameter(self, "preprocess", "off")
            cbc_set_parameter(self, "heur", "off")
            cbclib.Cbc_addCutCallback(
                self._model,
                cbc_cut_callback,
                "LazyConstraints".encode("utf-8"),
                ffi.NULL,
                1,
                atSol,
            )

        if self.emphasis == SearchEmphasis.FEASIBILITY:
            cbc_set_parameter(self, "passf", "50")
            cbc_set_parameter(self, "proximity", "on")
        if self.emphasis == SearchEmphasis.OPTIMALITY:
            cbc_set_parameter(self, "strong", "10")
            cbc_set_parameter(self, "trust", "20")
            cbc_set_parameter(self, "lagomory", "endonly")
            cbc_set_parameter(self, "latwomir", "endonly")

        if self.__pumpp != DEF_PUMPP:
            cbc_set_parameter(self, "passf", "{}".format(self.__pumpp))

        if self.model.cuts == 0:
            cbc_set_parameter(self, "cuts", "off")

        if self.model.cuts >= 1:
            cbc_set_parameter(self, "cuts", "on")
        if self.model.cuts >= 2:
            cbc_set_parameter(self, "lagomory", "endcleanroot")
            cbc_set_parameter(self, "latwomir", "endcleanroot")
            cbc_set_parameter(self, "passC", "-25")
        if self.model.cuts >= 3:
            cbc_set_parameter(self, "passC", "-35")
            cbc_set_parameter(self, "lift", "ifmove")

        if self.__threads >= 1:
            cbc_set_parameter(self, "timeM", "{}".format("elapsed"))
            Cbc_setIntParam(self._model, INT_PARAM_THREADS, self.__threads)
        elif self.__threads == -1:
            cbc_set_parameter(self, "threads", "{}".format(multip.cpu_count()))

        if self.model.preprocess == 0:
            cbc_set_parameter(self, "preprocess", "off")
        elif self.model.preprocess == 1:
            cbc_set_parameter(self, "preprocess", "sos")

        if self.model.cut_passes != -1:
            cbc_set_parameter(self, "passc", "{}".format(self.model.cut_passes))

        if self.model.clique == 0:
            cbc_set_parameter(self, "clique", "off")
        elif self.model.clique == 1:
            cbc_set_parameter(self, "clique", "forceon")

        cbc_set_parameter(self, "maxSavedSolutions", "10")

        if self.model.store_search_progress_log:
            cbclib.Cbc_addProgrCallback(self._model, cbc_progress_callback, ffi.NULL)

        if self.model.integer_tol >= 0.0:
            cbclib.Cbc_setDblParam(
                self._model, DBL_PARAM_INT_TOL, self.model.integer_tol
            )

        if self.model.infeas_tol >= 0.0:
            cbclib.Cbc_setPrimalTolerance(self._model, self.model.infeas_tol)

        if self.model.opt_tol >= 0.0:
            cbclib.Cbc_setDualTolerance(self._model, self.model.opt_tol)

        if self.model.lp_method == LP_Method.BARRIER:
            cbclib.Cbc_setLPmethod(self._model, cbclib.LPM_Barrier)
        elif self.model.lp_method == LP_Method.DUAL:
            cbclib.Cbc_setLPmethod(self._model, cbclib.LPM_Dual)
        elif self.model.lp_method == LP_Method.PRIMAL:
            cbclib.Cbc_setLPmethod(self._model, cbclib.LPM_Primal)
        else:
            cbclib.Cbc_setLPmethod(self._model, cbclib.LPM_Auto)

        cbclib.Cbc_setAllowableFractionGap(self._model, self.model.max_mip_gap)
        cbclib.Cbc_setAllowableGap(self._model, self.model.max_mip_gap_abs)
        cbclib.Cbc_setIntParam(self._model, INT_PARAM_RANDOM_SEED, self.model.seed)

        cbclib.Cbc_setIntParam(
            self._model, INT_PARAM_ROUND_INT_VARS, int(self.model.round_int_vars),
        )

        cbclib.Cbc_setIntParam(
            self._model, INT_PARAM_MAX_SAVED_SOLS, self.model.sol_pool_size
        )

        self.__clear_sol()
        cbclib.Cbc_solve(self._model)

        if cbclib.Cbc_isAbandoned(self._model):
            return OptimizationStatus.ERROR

        if cbclib.Cbc_isProvenOptimal(self._model):
            self.__x = cbclib.Cbc_getColSolution(self._model)
            self.__slack = cbclib.Cbc_getRowSlack(self._model)
            self.__obj_val = cbclib.Cbc_getObjValue(self._model) + self._objconst
            self.__obj_bound = self.__obj_val
            self.__num_solutions = 1

            if self.model.num_int == 0 and cbclib.Cbc_numberSOS(self._model) == 0:
                self.__rc = cbclib.Cbc_getReducedCost(self._model)
                self.__pi = cbclib.Cbc_getRowPrice(self._model)
                self.__slack = cbclib.Cbc_getRowSlack(self._model)
            else:
                self.__obj_bound = (
                    cbclib.Cbc_getBestPossibleObjValue(self._model) + self._objconst
                )
                self.__num_solutions = cbclib.Cbc_numberSavedSolutions(self._model)
            return OptimizationStatus.OPTIMAL

        if cbclib.Cbc_isProvenInfeasible(self._model):
            return OptimizationStatus.INFEASIBLE

        if cbclib.Cbc_isContinuousUnbounded(self._model):
            return OptimizationStatus.UNBOUNDED

        if cbclib.Cbc_getNumIntegers(self._model):
            self.__obj_bound = (
                cbclib.Cbc_getBestPossibleObjValue(self._model) + self._objconst
            )

            if cbclib.Cbc_bestSolution(self._model):
                self.__x = cbclib.Cbc_getColSolution(self._model)
                self.__slack = cbclib.Cbc_getRowSlack(self._model)
                self.__obj_val = cbclib.Cbc_getObjValue(self._model) + self._objconst
                self.__num_solutions = cbclib.Cbc_numberSavedSolutions(self._model)

                return OptimizationStatus.FEASIBLE

        return OptimizationStatus.NO_SOLUTION_FOUND

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
            raise ValueError(
                "Unknown sense: {}, use {} or {}".format(sense, MAXIMIZE, MINIMIZE)
            )

    def get_objective_value(self) -> numbers.Real:
        # return
        return self.__obj_val

    def get_status(self) -> OptimizationStatus:
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

        return OptimizationStatus.NO_SOLUTION_FOUND

    def get_log(self,) -> List[Tuple[numbers.Real, Tuple[numbers.Real, numbers.Real]]]:
        return self.__log

    def get_objective_bound(self) -> numbers.Real:
        return self.__obj_bound

    def var_get_x(self, var: Var) -> Optional[numbers.Real]:
        # model status is *already checked* Var x property
        # (returns None if no solution available)
        return self.__x[var.idx]

    def get_num_solutions(self) -> int:
        return self.__num_solutions

    def get_objective_value_i(self, i: int) -> numbers.Real:
        return cbclib.Cbc_savedSolutionObj(self._model, i) + self._objconst

    def var_get_xi(self, var: "Var", i: int) -> numbers.Real:
        # model status is *already checked* Var xi property
        # (returns None if no solution available)
        return cbclib.Cbc_savedSolution(self._model, i)[var.idx]

    def var_get_rc(self, var: Var) -> numbers.Real:
        # model status is *already checked* Var rc property
        # (returns None if no solution available)
        return self.__rc[var.idx]

    def var_get_lb(self, var: "Var") -> numbers.Real:
        return cbclib.Cbc_getColLB(self._model, var.idx)

    def var_set_lb(self, var: "Var", value: numbers.Real):
        cbclib.Cbc_setColLower(self._model, var.idx, value)

    def var_get_ub(self, var: "Var") -> numbers.Real:
        return cbclib.Cbc_getColUB(self._model, var.idx)

    def var_set_ub(self, var: "Var", value: numbers.Real):
        cbclib.Cbc_setColUpper(self._model, var.idx, value)

    def var_get_name(self, idx: int) -> str:
        namep = self.__name_space
        cbclib.Cbc_getColName(self._model, idx, namep, MAX_NAME_SIZE)
        return ffi.string(namep).decode("utf-8")

    def var_get_index(self, name: str) -> int:
        return cbclib.Cbc_getColNameIndex(self._model, name.encode("utf-8"))

    def constr_get_index(self, name: str) -> int:
        return cbclib.Cbc_getRowNameIndex(self._model, name.encode("utf-8"))

    def constr_get_rhs(self, idx: int) -> numbers.Real:
        return cbclib.Cbc_getRowRHS(self._model, idx)

    def constr_set_rhs(self, idx: int, rhs: numbers.Real):
        cbclib.Cbc_setRowRHS(self._model, idx, rhs)

    def var_get_obj(self, var: Var) -> numbers.Real:
        return cbclib.Cbc_getColObj(self._model, var.idx)

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
        if numnz == 0:
            return Column()

        cidx = cbclib.Cbc_getColIndices(self._model, var.idx)
        if cidx == ffi.NULL:
            raise ParameterNotAvailable("Error getting column indices'")
        ccoef = cbclib.Cbc_getColCoeffs(self._model, var.idx)

        return Column(
            [Constr(self.model, cidx[i]) for i in range(numnz)],
            [ccoef[i] for i in range(numnz)],
        )

    def add_constr(self, lin_expr: LinExpr, name: str = ""):
        # collecting linear expression data
        numnz = len(lin_expr.expr)

        if numnz > self.iidx_space:
            self.iidx_space = max(numnz, self.iidx_space * 2)
            self.iidx = ffi.new("int[%d]" % self.iidx_space)
            self.dvec = ffi.new("double[%d]" % self.iidx_space)

        # cind = self.iidx
        self.iidx = [var.idx for var in lin_expr.expr.keys()]

        # cind = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
        # cval = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])
        # cval = self.dvec
        self.dvec = [coef for coef in lin_expr.expr.values()]

        # constraint sense and rhs
        sense = lin_expr.sense.encode("utf-8")
        rhs = -lin_expr.const

        namestr = name.encode("utf-8")
        mp = self._model
        cbclib.Cbc_addRow(mp, namestr, numnz, self.iidx, self.dvec, sense, rhs)

    def add_lazy_constr(self: "Solver", lin_expr: LinExpr):
        # collecting linear expression data
        numnz = len(lin_expr.expr)

        cind = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
        cval = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])

        # constraint sense and rhs
        sense = lin_expr.sense.encode("utf-8")
        rhs = -lin_expr.const

        mp = self._model
        cbclib.Cbc_addLazyConstraint(mp, numnz, cind, cval, sense, rhs)

    def add_sos(self, sos: List[Tuple["Var", numbers.Real]], sos_type: int):
        starts = ffi.new("int[]", [0, len(sos)])
        idx = ffi.new("int[]", [v.idx for (v, f) in sos])
        w = ffi.new("double[]", [f for (v, f) in sos])
        cbclib.Cbc_addSOS(self._model, 1, starts, idx, w, sos_type)

    def add_cut(self, lin_expr: LinExpr):
        global cut_idx
        name = "cut{}".format(cut_idx)
        self.add_constr(lin_expr, name)

    def write(self, file_path: str):
        fpstr = file_path.encode("utf-8")
        if ".mps" in file_path.lower():
            cbclib.Cbc_writeMps(self._model, fpstr)
        elif ".lp" in file_path.lower():
            cbclib.Cbc_writeLp(self._model, fpstr)
        else:
            raise ValueError(
                "Enter a valid extension (.lp or .mps) \
                to indicate the file format"
            )

    def read(self, file_path: str) -> None:
        if not isfile(file_path):
            raise FileNotFoundError("File {} does not exists".format(file_path))

        if file_path.lower().endswith(".gz") and cbclib.Cbc_supportsGzip() == CHAR_ZERO:
            raise MipBaseException("CBC not compiled with gzip support")
        if (
            file_path.lower().endswith(".bz2")
            and cbclib.Cbc_supportsBzip2() == CHAR_ZERO
        ):
            raise MipBaseException("CBC not compiled with bzip2 support")

        fpstr = file_path.encode("utf-8")
        if ".mps" in file_path.lower():
            cbclib.Cbc_readMps(self._model, fpstr)
        elif ".lp" in file_path.lower():
            cbclib.Cbc_readLp(self._model, fpstr)
        else:
            raise ValueError(
                "Enter a valid extension (.lp or .mps) \
                to indicate the file format"
            )

    def set_start(self, start: List[Tuple[Var, numbers.Real]]) -> None:
        n = len(start)
        dv = ffi.new("double[]", [start[i][1] for i in range(n)])
        keep_alive_str = [
            ffi.new("char[]", str.encode(start[i][0].name)) for i in range(n)
        ]
        var_names = ffi.new("char *[]", keep_alive_str)
        mdl = self._model
        cbclib.Cbc_setMIPStart(mdl, n, var_names, dv)

    def num_cols(self) -> int:
        return cbclib.Cbc_getNumCols(self._model)

    def num_int(self) -> int:
        return cbclib.Cbc_getNumIntegers(self._model)

    def num_rows(self) -> int:
        return cbclib.Cbc_getNumRows(self._model)

    def num_nz(self) -> int:
        return cbclib.Cbc_getNumElements(self._model)

    def get_cutoff(self) -> numbers.Real:
        return cbclib.Cbc_getCutoff(self._model)

    def set_cutoff(self, cutoff: numbers.Real):
        cbclib.Cbc_setCutoff(self._model, cutoff)

    def get_mip_gap_abs(self) -> numbers.Real:
        return cbclib.Cbc_getAllowableGap(self._model)

    def set_mip_gap_abs(self, allowable_gap: numbers.Real):
        cbclib.Cbc_setAllowableGap(self._model, allowable_gap)

    def get_mip_gap(self) -> numbers.Real:
        return cbclib.Cbc_getAllowableFractionGap(self._model)

    def set_mip_gap(self, allowable_ratio_gap: numbers.Real):
        cbclib.Cbc_setAllowableFractionGap(self._model, allowable_ratio_gap)

    def constr_get_expr(self, constr: Constr) -> LinExpr:
        numnz = cbclib.Cbc_getRowNz(self._model, constr.idx)

        ridx = cbclib.Cbc_getRowIndices(self._model, constr.idx)
        if ridx == ffi.NULL:
            raise ParameterNotAvailable("Error getting row indices.")
        rcoef = cbclib.Cbc_getRowCoeffs(self._model, constr.idx)
        if rcoef == ffi.NULL:
            raise ParameterNotAvailable("Error getting row coefficients.")

        rhs = cbclib.Cbc_getRowRHS(self._model, constr.idx)
        rsense = cbclib.Cbc_getRowSense(self._model, constr.idx).decode("utf-8").upper()
        sense = ""
        if rsense == "E":
            sense = EQUAL
        elif rsense == "L":
            sense = LESS_OR_EQUAL
        elif rsense == "G":
            sense = GREATER_OR_EQUAL
        else:
            raise ValueError("Unknow sense: {}".format(rsense))

        expr = LinExpr(const=-rhs, sense=sense)
        for i in range(numnz):
            expr.add_var(self.model.vars[ridx[i]], rcoef[i])

        return expr

    def constr_get_name(self, idx: int) -> str:
        namep = self.__name_space
        cbclib.Cbc_getRowName(self._model, idx, namep, MAX_NAME_SIZE)
        return ffi.string(namep).decode("utf-8")

    def set_processing_limits(
        self,
        max_time: numbers.Real = INF,
        max_nodes: int = maxsize,
        max_sol: int = maxsize,
    ):
        if max_time != INF:
            cbc_set_parameter(self, "timeMode", "elapsed")
            self.set_max_seconds(max_time)
        if max_nodes != INF:
            self.set_max_nodes(max_nodes)
        if max_sol != INF:
            self.set_max_solutions(max_sol)

    def get_emphasis(self) -> SearchEmphasis:
        return self.emphasis

    def set_emphasis(self, emph: SearchEmphasis):
        self.emphasis = emph

    def set_num_threads(self, threads: int):
        self.__threads = threads

    def remove_constrs(self, constrs: List[int]):
        idx = ffi.new("int[]", constrs)
        cbclib.Cbc_deleteRows(self._model, len(constrs), idx)

    def remove_vars(self, varsList: List[int]):
        idx = ffi.new("int[]", varsList)
        cbclib.Cbc_deleteCols(self._model, len(varsList), idx)

    def __del__(self):
        cbclib.Cbc_deleteModel(self._model)

    def get_problem_name(self) -> str:
        namep = self.__name_space
        cbclib.Cbc_problemName(self._model, MAX_NAME_SIZE, namep)
        return ffi.string(namep).decode("utf-8")

    def set_problem_name(self, name: str):
        cbclib.Cbc_setProblemName(self._model, name.encode("utf-8"))

    def get_pump_passes(self) -> int:
        return self.__pumpp

    def set_pump_passes(self, passes: int):
        self.__pumpp = passes

    def constr_get_pi(self, constr: Constr) -> Optional[numbers.Real]:
        return self.__pi[constr.idx]

    def constr_get_slack(self, constr: Constr) -> Optional[numbers.Real]:
        return self.__slack[constr.idx]


class ModelOsi(Model):
    def __init__(self, osi_ptr):
        # initializing variables with default values
        self.solver_name = "osi"
        existing_solver = osi_ptr != ffi.NULL

        self.solver = SolverOsi(self, osi_ptr)

        # list of constraints and variables
        self.constrs = VConstrList(self)
        self.vars = VVarList(self)

        # if a fractional solution is being processed
        self.fractional = True

        if existing_solver:
            self._status = self.solver.get_status()
        else:
            self._status = OptimizationStatus.LOADED

        # initializing additional control variables
        self.__cuts = -1
        self.__cut_passes = -1
        self.__clique = -1
        self.__preprocess = -1
        self.__constrs_generator = None
        self.__lazy_constrs_generator = None
        self.__start = None
        self.__threads = 0
        self.__n_cols = 0
        self.__n_rows = 0
        self.__gap = INF
        self.__store_search_progress_log = False

    def add_constr(self, lin_expr: LinExpr, name: str = "") -> "Constr":
        if self.fractional:
            self.add_cut(lin_expr)
            return None

        self.add_lazy_constr(lin_expr)
        return None


class SolverOsi(Solver):
    """Interface for the OsiSolverInterface, the generic solver interface of
    COIN-OR. This solver has a restricted functionality (comparing to
    SolverCbc) and it is used mainly in callbacks where only the pre-processed
    model is available"""

    def __init__(self, model: Model, osi_ptr=ffi.NULL):
        super().__init__(model)

        self._objconst = 0.0

        # pre-allocate temporary space to query names
        self.__name_space = ffi.new("char[{}]".format(MAX_NAME_SIZE))
        # in cut generation
        self.__name_spacec = ffi.new("char[{}]".format(MAX_NAME_SIZE))

        if osi_ptr != ffi.NULL:
            self.osi = osi_ptr
            self.owns_solver = False
        else:
            self.owns_solver = True
            self.osi = cbclib.Osi_newSolver()
        self.__relaxed = False

        # name indexes, created if necessary
        self.colNames = None
        self.rowNames = None
        self._objconst = 0.0
        self.osi_cutsp = ffi.NULL
        self.__x = EmptyVarSol(model)
        self.__rc = EmptyVarSol(model)
        self.__pi = EmptyRowSol(model)
        self.__obj_val = None

        if cbclib.Osi_isProvenOptimal(self.osi):
            self.__x = cbclib.Osi_getColSolution(self.osi)
            self.__rc = cbclib.Osi_getReducedCost(self.osi)
            self.__pi = cbclib.Osi_getRowPrice(self.osi)
            self.__obj_val = cbclib.Osi_getObjValue(self.osi)

    def __clear_sol(self: "SolverOsi"):
        self.__x = EmptyVarSol(self.model)
        self.__rc = EmptyVarSol(self.model)
        self.__pi = EmptyRowSol(self.model)
        self.__obj_val = None

    def __del__(self):
        if self.owns_solver:
            cbclib.Osi_deleteSolver(self.osi)

    def add_var(
        self,
        name: str = "",
        obj: numbers.Real = 0,
        lb: numbers.Real = 0,
        ub: numbers.Real = INF,
        var_type: str = CONTINUOUS,
        column: "Column" = None,
    ):
        # collecting column data
        if column is None:
            vind = ffi.NULL
            vval = ffi.NULL
            numnz = 0
        else:
            vind = ffi.new("int[]", [c.idx for c in column.constrs])
            vval = ffi.new("double[]", [coef for coef in column.coeffs])
            numnz = len(column.constrs)

        isInt = (
            CHAR_ONE if var_type.upper() == "B" or var_type.upper() == "I" else CHAR_ZERO
        )
        cbclib.Osi_addCol(
            self.osi, name.encode("utf-8"), lb, ub, obj, isInt, numnz, vind, vval,
        )

    def add_constr(self, lin_expr: "LinExpr", name: str = ""):
        # collecting linear expression data
        numnz = len(lin_expr.expr)

        cind = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
        cval = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])

        # constraint sense and rhs
        sense = lin_expr.sense.encode("utf-8")
        rhs = -lin_expr.const

        namestr = name.encode("utf-8")
        mp = self.osi
        cbclib.Osi_addRow(mp, namestr, numnz, cind, cval, sense, rhs)

    def add_cut(self, lin_expr: LinExpr):
        if self.osi_cutsp != ffi.NULL:
            if lin_expr.violation < 1e-5:
                return

            numnz = len(lin_expr.expr)

            cind = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
            cval = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])

            # constraint sense and rhs
            sense = lin_expr.sense.encode("utf-8")
            rhs = -lin_expr.const

            OsiCuts_addGlobalRowCut(self.osi_cutsp, numnz, cind, cval, sense, rhs)
        else:
            global cut_idx
            name = "cut{}".format(cut_idx)
            self.add_constr(lin_expr, name)

    def add_lazy_constr(self, lin_expr: LinExpr):
        if self.osi_cutsp != ffi.NULL:
            # checking if violated
            if lin_expr.violation < 1e-5:
                return

            numnz = len(lin_expr.expr)

            cind = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
            cval = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])

            # constraint sense and rhs
            sense = lin_expr.sense.encode("utf-8")
            rhs = -lin_expr.const

            OsiCuts_addGlobalRowCut(self.osi_cutsp, numnz, cind, cval, sense, rhs)
        else:
            global cut_idx
            name = "cut{}".format(cut_idx)
            self.add_constr(lin_expr, name)

    def get_objective_bound(self) -> numbers.Real:
        raise NotImplementedError("Not available in OsiSolver")

    def get_objective(self) -> LinExpr:
        obj = cbclib.Osi_getObjCoefficients(self.osi)
        if obj == ffi.NULL:
            raise ParameterNotAvailable("Error getting objective function coefficients")
        return (
            xsum(
                obj[j] * self.model.vars[j]
                for j in range(self.num_cols())
                if abs(obj[j]) >= 1e-15
            )
            + self._objconst
        )

    def get_objective_const(self) -> numbers.Real:
        return self._objconst

    def relax(self):
        self.__relaxed = True

    def optimize(self) -> OptimizationStatus:
        if self.__relaxed or self.num_int() == 0:
            # linear optimization
            if cbclib.Osi_isProvenOptimal(self.osi):
                cbclib.Osi_resolve(self.osi)
            else:
                cbclib.Osi_initialSolve(self.osi)
        else:
            cbclib.Osi_branchAndBound(self.osi)

        if cbclib.Osi_isProvenOptimal(self.osi):
            self.__x = cbclib.Osi_getColSolution(self.osi)
            self.__rc = cbclib.Osi_getReducedCost(self.osi)
            self.__pi = cbclib.Osi_getRowPrice(self.osi)
            self.__obj_val = cbclib.Osi_getObjValue(self.osi)

        return self.get_status()

    def get_status(self) -> OptimizationStatus:
        if cbclib.Osi_isProvenOptimal(self.osi):
            return OptimizationStatus.OPTIMAL

        if cbclib.Osi_isProvenPrimalInfeasible(
            self.osi
        ) or cbclib.Osi_isProvenDualInfeasible(self.osi):
            return OptimizationStatus.INFEASIBLE
        elif cbclib.Osi_isAbandoned(self.osi):
            return OptimizationStatus.ERROR
        return OptimizationStatus.LOADED

    def get_objective_value(self) -> numbers.Real:
        return self.__obj_val

    def get_log(self,) -> List[Tuple[numbers.Real, Tuple[numbers.Real, numbers.Real]]]:
        return []

    def get_objective_value_i(self, i: int) -> numbers.Real:
        raise NotImplementedError("Not available in OsiSolver")

    def get_num_solutions(self) -> int:
        if cbclib.Osi_isProvenOptimal(self.osi):
            return 1

        return 0

    def get_objective_sense(self) -> str:
        objs = cbclib.Osi_getObjSense(self.osi)
        if objs <= -0.5:
            return MAXIMIZE

        return MINIMIZE

    def set_objective_sense(self, sense: str):
        if sense.strip().upper() == MAXIMIZE.strip().upper():
            cbclib.Osi_setObjSense(self.osi, -1.0)
        elif sense.strip().upper() == MINIMIZE.strip().upper():
            cbclib.Osi_setObjSense(self.osi, 1.0)
        else:
            raise ValueError(
                "Unknown sense: {}, use {} or {}".format(sense, MAXIMIZE, MINIMIZE)
            )

    def set_start(self, start: List[Tuple["Var", numbers.Real]]):
        raise NotImplementedError("MIPstart not available in OsiSolver")

    def set_objective(self, lin_expr: "LinExpr", sense: str = ""):
        # collecting variable coefficients
        for var, coeff in lin_expr.expr.items():
            cbclib.Osi_setObjCoeff(self.osi, var.idx, coeff)

        # objective function constant
        self._objconst = lin_expr.const

        # setting objective sense
        if sense == MAXIMIZE:
            cbclib.Osi_setObjSense(self.osi, -1.0)
        elif sense == MINIMIZE:
            cbclib.Osi_setObjSense(self.osi, 1.0)

    def set_objective_const(self, const: numbers.Real):
        raise NotImplementedError("Still not implemented in OsiSolver")

    def set_processing_limits(
        self,
        max_time: numbers.Real = INF,
        max_nodes: int = maxsize,
        max_sol: int = maxsize,
    ):
        raise NotImplementedError("Not available in OsiSolver")

    def get_max_seconds(self) -> numbers.Real:
        raise NotImplementedError("Not available in OsiSolver")

    def set_max_seconds(self, max_seconds: numbers.Real):
        raise NotImplementedError("Not available in OsiSolver")

    def get_max_solutions(self) -> int:
        raise NotImplementedError("Not available in OsiSolver")

    def set_max_solutions(self, max_solutions: int):
        raise NotImplementedError("Not available in OsiSolver")

    def get_pump_passes(self) -> int:
        raise NotImplementedError("Not available in OsiSolver")

    def set_pump_passes(self, passes: int):
        raise NotImplementedError("Not available in OsiSolver")

    def get_max_nodes(self) -> int:
        raise NotImplementedError("Not available in OsiSolver")

    def set_max_nodes(self, max_nodes: int):
        raise NotImplementedError("Not available in OsiSolver")

    def set_num_threads(self, threads: int):
        raise NotImplementedError("Not available in OsiSolver")

    def write(self, file_path: str):
        raise NotImplementedError("Not available in OsiSolver")

    def read(self, file_path: str):
        raise NotImplementedError("Not available in OsiSolver")

    def num_cols(self) -> int:
        return cbclib.Osi_getNumCols(self.osi)

    def num_rows(self) -> int:
        return cbclib.Osi_getNumRows(self.osi)

    def num_nz(self) -> int:
        return cbclib.Osi_getNumElements(self.osi)

    def num_int(self) -> int:
        return cbclib.Osi_getNumIntegers(self.osi)

    def get_emphasis(self) -> SearchEmphasis:
        raise NotImplementedError("Not available in OsiSolver")

    def set_emphasis(self, emph: SearchEmphasis):
        raise NotImplementedError("Not available in OsiSolver")

    def get_cutoff(self) -> numbers.Real:
        raise NotImplementedError("Not available in OsiSolver")

    def set_cutoff(self, cutoff: numbers.Real):
        raise NotImplementedError("Not available in OsiSolver")

    def get_mip_gap_abs(self) -> numbers.Real:
        raise NotImplementedError("Not available in OsiSolver")

    def set_mip_gap_abs(self, mip_gap_abs: numbers.Real):
        raise NotImplementedError("Not available in OsiSolver")

    def get_mip_gap(self) -> numbers.Real:
        raise NotImplementedError("Not available in OsiSolver")

    def set_mip_gap(self, mip_gap: numbers.Real):
        raise NotImplementedError("Not available in OsiSolver")

    def get_verbose(self) -> int:
        raise NotImplementedError("Not available in OsiSolver")

    def set_verbose(self, verbose: int):
        raise NotImplementedError("Not available in OsiSolver")

    # Constraint-related getters/setters
    def constr_get_expr(self, constr: Constr) -> LinExpr:
        numnz = cbclib.Osi_getRowNz(self.osi, constr.idx)

        ridx = cbclib.Osi_getRowIndices(self.osi, constr.idx)
        if ridx == ffi.NULL:
            raise ParameterNotAvailable("Error getting row indices.")
        rcoef = cbclib.Osi_getRowCoeffs(self.osi, constr.idx)
        if rcoef == ffi.NULL:
            raise ParameterNotAvailable("Error getting row coefficients.")

        rhs = cbclib.Osi_getRowRHS(self.osi, constr.idx)
        rsense = cbclib.Osi_getRowSense(self.osi, constr.idx).decode("utf-8").upper()
        sense = ""
        if rsense == "E":
            sense = EQUAL
        elif rsense == "L":
            sense = LESS_OR_EQUAL
        elif rsense == "G":
            sense = GREATER_OR_EQUAL
        else:
            raise ValueError("Unknow sense: {}".format(rsense))

        expr = LinExpr(const=-rhs, sense=sense)
        for i in range(numnz):
            expr.add_var(self.model.vars[ridx[i]], rcoef[i])

        return expr

    def constr_set_expr(self, constr: Constr, value: LinExpr) -> LinExpr:
        raise NotImplementedError("Not available in OsiSolver")

    def constr_get_name(self, idx: int) -> str:
        namep = self.__name_space
        cbclib.Osi_getRowName(self.osi, idx, namep, MAX_NAME_SIZE)
        return ffi.string(namep).decode("utf-8")

    def remove_constrs(self, constrsList: List[int]):
        raise NotImplementedError("Not available in OsiSolver")

    def constr_get_index(self, name: str) -> int:
        if self.rowNames is None:
            self.rowNames = {}
            for i in range(self.num_rows()):
                self.rowNames[self.constr_get_name(i)] = i

        if name in self.rowNames:
            return self.rowNames[name]

        return -1

    def constr_get_pi(self, constr: Constr) -> Optional[numbers.Real]:
        return self.__pi[constr.idx]

    def constr_get_slack(self, constr: Constr) -> Optional[float]:
        if self.model.status not in [
            OptimizationStatus.OPTIMAL,
            OptimizationStatus.FEASIBLE,
        ]:
            return None
        pac = cbclib.Osi_getRowActivity(self.osi)
        if pac == ffi.NULL:
            return None
        rhs = cbclib.Osi_getRowRHS(self.osi, constr.idx)
        activity = pac[constr.idx]

        sense = cbclib.Osi_getRowSense(self.osi, constr.idx).decode("utf-8").upper()

        if sense in "<L":
            return rhs - activity
        if sense in ">G":
            return activity - rhs
        if sense in "=E":
            return abs(activity - rhs)

        return None

    # Variable-related getters/setters
    def var_get_lb(self, var: "Var") -> numbers.Real:
        x = cbclib.Osi_getColLower(self.osi)
        return x[var.idx]

    def var_set_lb(self, var: "Var", value: numbers.Real):
        cbclib.Osi_setColLower(self.osi, var.idx, value)

    def var_get_ub(self, var: "Var") -> numbers.Real:
        x = cbclib.Osi_getColUpper(self.osi)
        return x[var.idx]

    def var_set_ub(self, var: "Var", value: numbers.Real):
        cbclib.Osi_setColUpper(self.osi, var.idx, value)

    def var_get_obj(self, var: "Var") -> numbers.Real:
        obj = cbclib.Osi_getObjCoefficients(self.osi)
        if obj == ffi.NULL:
            raise ParameterNotAvailable("Error getting objective function coefficients")
        return obj[var.idx]

    def var_set_obj(self, var: "Var", value: numbers.Real):
        cbclib.Osi_setObjCoef(self.osi, var.idx, value)

    def var_get_var_type(self, var: "Var") -> str:
        isInt = cbclib.Osi_isInteger(self.osi, var.idx)
        if isInt:
            lb = self.var_get_lb(var)
            ub = self.var_get_ub(var)
            if abs(lb) <= 1e-15 and abs(ub - 1.0) <= 1e-15:
                return BINARY

            return INTEGER

        return CONTINUOUS

    def var_set_var_type(self, var: "Var", value: str):
        cv = var.var_type
        if value == cv:
            return
        if cv == CONTINUOUS:
            if value in (INTEGER, BINARY):
                cbclib.Osi_setInteger(self.osi, var.idx)
        else:
            if value == CONTINUOUS:
                cbclib.Osi_setContinuous(self.osi, var.idx)
        if value == BINARY:
            # checking bounds
            if var.lb != 0.0:
                var.lb = 0.0
            if var.ub != 1.0:
                var.ub = 1.0

    def var_get_column(self, var: "Var") -> Column:
        numnz = cbclib.Cbc_getColNz(self.osi, var.idx)

        cidx = cbclib.Cbc_getColIndices(self.osi, var.idx)
        if cidx == ffi.NULL:
            raise ParameterNotAvailable("Error getting column indices'")
        ccoef = cbclib.Cbc_getColCoeffs(self.osi, var.idx)

        col = Column()

        for i in range(numnz):
            col.constrs.append(Constr(self, cidx[i]))
            col.coeffs.append(ccoef[i])

        return col

    def var_set_column(self, var: "Var", value: Column):
        raise NotImplementedError("Not available in OsiSolver")

    def var_get_rc(self, var: "Var") -> numbers.Real:
        return self.__rc[var.idx]

    def var_get_x(self, var: "Var") -> numbers.Real:
        return self.__x[var.idx]

    def var_get_xi(self, var: "Var", i: int) -> numbers.Real:
        raise NotImplementedError("Solution pool not supported in OsiSolver")

    def var_get_name(self, idx: int) -> str:
        namep = self.__name_space
        cbclib.Osi_getColName(self.osi, idx, namep, MAX_NAME_SIZE)
        return ffi.string(namep).decode("utf-8")

    def remove_vars(self, varsList: List[int]):
        raise NotImplementedError("Not supported in OsiSolver")

    def var_get_index(self, name: str) -> int:
        if self.colNames is None:
            self.colNames = {}
            for i in range(self.num_cols()):
                self.colNames[self.var_get_name(i)] = i

        if name in self.colNames:
            return self.colNames[name]

        return -1

    def get_problem_name(self) -> str:
        raise NotImplementedError("Not supported in OsiSolver")

    def set_problem_name(self, name: str):
        raise NotImplementedError("Not supported in OsiSolver")


# vim: ts=4 sw=4 et
