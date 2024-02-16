#
# This file implements python-mip interface for the Cardinal Optimizer
#

import os
import sys
import math
import numbers

from glob import glob
from cffi import FFI
from typing import List, Tuple

from mip.exceptions import (
  ParameterNotAvailable,
  SolutionNotAvailable,
  ProgrammingError,
  InterfacingError,
)
from mip import (
  Model,
  Column,
  Var,
  LinExpr,
  Constr,
  Solver,
  VConstrList,
  VVarList,
  MAXIMIZE,
  MINIMIZE,
  CONTINUOUS,
  INTEGER,
  BINARY,
  EQUAL,
  LESS_OR_EQUAL,
  GREATER_OR_EQUAL,
  OptimizationStatus,
  SearchEmphasis,
  LP_Method,
  xsum
)
from mip.lists import EmptyVarSol, EmptyRowSol

import mip

try:
  libpath = None
  libfile = None

  if 'COPT_HOME' in os.environ:
    copthome = os.getenv('COPT_HOME')

    if sys.platform == 'win32':
      libfile = glob(os.path.join(copthome, 'bin', 'copt.dll'))
    elif sys.platform == 'linux':
      libfile = glob(os.path.join(copthome, 'lib', 'libcopt.so'))
    elif sys.platform == 'darwin':
      libfile = glob(os.path.join(copthome, 'lib', 'libcopt.dylib'))
    else:
      raise InterfacingError("Unsupported operating system")

    if libfile:
      libpath = libfile[0]

  if libpath is None:
    raise FileNotFoundError("'COPT_HOME' environment variable not set correctly, "
                            "please refer to the installation guide for details.")

  ffi = FFI()
  coptlib = ffi.dlopen(libpath)
except Exception:
  raise ImportError("Failed to import Cardinal Optimizer")

CData         = ffi.CData
INF           = float('inf')

COPT_BUFFSIZE = 1000
COPT          = 'COPT'

ffi.cdef("""
typedef struct copt_env_s  copt_env;
typedef struct copt_prob_s copt_prob;

int COPT_CreateEnv(copt_env **p_env);
int COPT_DeleteEnv(copt_env **p_env);

int COPT_CreateProb(copt_env *env, copt_prob **p_prob);
int COPT_DeleteProb(copt_prob **p_prob);

int COPT_LoadProb(copt_prob *prob,
    int               nCol,
    int               nRow,
    int               iObjSense,
    double            dObjConst,
    const double      *colObj,
    const int         *colMatBeg,
    const int         *colMatCnt,
    const int         *colMatIdx,
    const double      *colMatElem,
    const char        *colType,
    const double      *colLower,
    const double      *colUpper,
    const char        *rowSense,
    const double      *rowBound,
    const double      *rowUpper,
    char const *const *colNames,
    char const *const *rowNames);

int COPT_AddCol(copt_prob *prob,
    double            dColObj,
    int               nColMatCnt,
    const int         *colMatIdx,
    const double      *colMatElem,
    char              cColType,
    double            dColLower,
    double            dColUpper,
    const char        *colName);

int COPT_AddRow(copt_prob *prob,
    int               nRowMatCnt,
    const int         *rowMatIdx,
    const double      *rowMatElem,
    char              cRowSense,
    double            dRowBound,
    double            dRowUpper,
    const char        *rowName);

int COPT_AddCols(copt_prob *prob,
    int               nAddCol,
    const double      *colObj,
    const int         *colMatBeg,
    const int         *colMatCnt,
    const int         *colMatIdx,
    const double      *colMatElem,
    const char        *colType,
    const double      *colLower,
    const double      *colUpper,
    char const *const *colNames);

int COPT_AddRows(copt_prob *prob,
    int               nAddRow,
    const int         *rowMatBeg,
    const int         *rowMatCnt,
    const int         *rowMatIdx,
    const double      *rowMatElem,
    const char        *rowSense,
    const double      *rowBound,
    const double      *rowUpper,
    char const *const *rowNames);

int COPT_AddLazyConstr(copt_prob *prob,
    int               nRowMatCnt,
    const int         *rowMatIdx,
    const double      *rowMatElem,
    char              cRowSense,
    double            dRowBound,
    double            dRowUpper,
    const char        *rowName);

int COPT_AddUserCut(copt_prob *prob,
    int               nRowMatCnt,
    const int         *rowMatIdx,
    const double      *rowMatElem,
    char              cRowSense,
    double            dRowBound,
    double            dRowUpper,
    const char        *rowName);

int COPT_AddSOSs(copt_prob *prob,
    int               nAddSOS,
    const int         *sosType,
    const int         *sosMatBeg,
    const int         *sosMatCnt,
    const int         *sosMatIdx,
    const double      *sosMatWt);

int COPT_AddIndicator(copt_prob *prob,
    int               binColIdx,
    int               binColVal,
    int               nRowMatCnt,
    const int         *rowMatIdx,
    const double      *rowMatElem,
    char              cRowSense,
    double            dRowBound);

int COPT_GetCols(copt_prob *prob,
    int               nCol,
    const int         *list,
    int               *colMatBeg,
    int               *colMatCnt,
    int               *colMatIdx,
    double            *colMatElem,
    int               nElemSize,
    int               *pReqSize);

int COPT_GetRows(copt_prob *prob,
    int               nRow,
    const int         *list,
    int               *rowMatBeg,
    int               *rowMatCnt,
    int               *rowMatIdx,
    double            *rowMatElem,
    int               nElemSize,
    int               *pReqSize);

int COPT_GetSOSs(copt_prob *prob,
    int               nSos,
    const int         *list,
    int               *sosType,
    int               *sosMatBeg,
    int               *sosMatCnt,
    int               *sosMatIdx,
    double            *sosMatWt,
    int               nElemSize,
    int               *pReqSize);

int COPT_GetIndicator(copt_prob *prob,
    int               rowIdx,
    int               *binColIdx,
    int               *binColVal,
    int               *nRowMatCnt,
    int               *rowMatIdx,
    double            *rowMatElem,
    char              *cRowSense,
    double            *dRowBound,
    int               nElemSize,
    int               *pReqSize);

int COPT_GetElem(copt_prob *prob, int iCol, int iRow, double *p_elem);
int COPT_SetElem(copt_prob *prob, int iCol, int iRow, double newElem);

int COPT_DelCols(copt_prob *prob, int num, const int *list);
int COPT_DelRows(copt_prob *prob, int num, const int *list);
int COPT_DelSOSs(copt_prob *prob, int num, const int *list);
int COPT_DelIndicators(copt_prob *prob, int num, const int *list);

int COPT_SetObjSense(copt_prob *prob, int iObjSense);
int COPT_SetObjConst(copt_prob *prob, double dObjConst);

int COPT_SetColObj(copt_prob *prob, int num, const int *list, const double *obj);
int COPT_SetColType(copt_prob *prob, int num, const int *list, const char *type);
int COPT_SetColLower(copt_prob *prob, int num, const int *list, const double *lower);
int COPT_SetColUpper(copt_prob *prob, int num, const int *list, const double *upper);
int COPT_SetColNames(copt_prob *prob, int num, const int *list, char const *const *names);

int COPT_SetRowLower(copt_prob *prob, int num, const int *list, const double *lower);
int COPT_SetRowUpper(copt_prob *prob, int num, const int *list, const double *upper);
int COPT_SetRowNames(copt_prob *prob, int num, const int *list, char const *const *names);

int COPT_ReplaceColObj(copt_prob *prob, int num, const int *list, const double *obj);

int COPT_ReadMps(copt_prob *prob, const char *mpsfilename);
int COPT_ReadLp(copt_prob *prob, const char *lpfilename);
int COPT_ReadCbf(copt_prob *prob, const char *cbffilename);
int COPT_ReadBin(copt_prob *prob, const char *binfilename);
int COPT_ReadSol(copt_prob *prob, const char *solfilename);
int COPT_ReadBasis(copt_prob *prob, const char *basfilename);
int COPT_ReadMst(copt_prob *prob, const char *mstfilename);
int COPT_ReadParam(copt_prob *prob, const char *parfilename);

int COPT_WriteMps(copt_prob *prob, const char *mpsfilename);
int COPT_WriteLp(copt_prob *prob, const char *lpfilename);
int COPT_WriteCbf(copt_prob *prob, const char *cbffilename);
int COPT_WriteBin(copt_prob *prob, const char *binfilename);
int COPT_WriteSol(copt_prob *prob, const char *solfilename);
int COPT_WriteBasis(copt_prob *prob, const char *basfilename);
int COPT_WriteMst(copt_prob *prob, const char *mstfilename);
int COPT_WriteParam(copt_prob *prob, const char *parfilename);

int COPT_AddMipStart(copt_prob *prob, int num, const int *list, double *colVal);

int COPT_SolveLp(copt_prob *prob);
int COPT_Solve(copt_prob *prob);

int COPT_GetSolution(copt_prob *prob, double *colVal);
int COPT_GetLpSolution(copt_prob *prob, double *value, double *slack, double *rowDual, double *redCost);
int COPT_SetLpSolution(copt_prob *prob, const double *value, const double *slack, const double *rowDual, const double *redCost);
int COPT_GetBasis(copt_prob *prob, int *colBasis, int *rowBasis);
int COPT_SetBasis(copt_prob *prob, const int *colBasis, const int *rowBasis);
int COPT_SetSlackBasis(copt_prob *prob);

int COPT_GetPoolObjVal(copt_prob *prob, int iSol, double *p_objVal);
int COPT_GetPoolSolution(copt_prob *prob, int iSol, int num, const int *list, double *colVal);

int COPT_SetIntParam(copt_prob *prob, const char *paramName, int intParam);
int COPT_GetIntParam(copt_prob *prob, const char *paramName, int *p_intParam);
int COPT_GetIntParamDef(copt_prob *prob, const char *paramName, int *p_intParam);

int COPT_SetDblParam(copt_prob *prob, const char *paramName, double dblParam);
int COPT_GetDblParam(copt_prob *prob, const char *paramName, double *p_dblParam);
int COPT_GetDblParamDef(copt_prob *prob, const char *paramName, double *p_dblParam);

int COPT_ResetParam(copt_prob *prob);
int COPT_Reset(copt_prob *prob, int iClearAll);

int COPT_GetIntAttr(copt_prob *prob, const char *attrName, int *p_intAttr);
int COPT_GetDblAttr(copt_prob *prob, const char *attrName, double *p_dblAttr);

int COPT_GetColIdx(copt_prob *prob, const char *colName, int *p_iCol);
int COPT_GetRowIdx(copt_prob *prob, const char *rowName, int *p_iRow);
int COPT_GetColInfo(copt_prob *prob, const char *infoName, int num, const int *list, double *info);
int COPT_GetRowInfo(copt_prob *prob, const char *infoName, int num, const int *list, double *info);

int COPT_GetColType(copt_prob *prob, int num, const int *list, char *type);
int COPT_GetColBasis(copt_prob *prob, int num, const int *list, int *colBasis);
int COPT_GetRowBasis(copt_prob *prob, int num, const int *list, int *rowBasis);

int COPT_GetColName(copt_prob *prob, int iCol, char *buff, int buffSize, int *pReqSize);
int COPT_GetRowName(copt_prob *prob, int iRow, char *buff, int buffSize, int *pReqSize);

int COPT_SetCallback(copt_prob *prob, 
                      int (*cb)(copt_prob *prob, void *cbdata, int cbctx, void *userdata),
                      int cbctx, void *userdata);
int COPT_GetCallbackInfo(void *cbdata, const char *cbinfo, void *p_val);
int COPT_AddCallbackSolution(void *cbdata, const double *sol, double *p_objval);
int COPT_AddCallbackUserCut(void *cbdata,
    int               nRowMatCnt,
    const int         *rowMatIdx,
    const double      *rowMatElem,
    char              cRowSense,
    double            dRowRhs);
int COPT_AddCallbackLazyConstr(void *cbdata,
    int               nRowMatCnt,
    const int         *rowMatIdx,
    const double      *rowMatElem,
    char              cRowSense,
    double            dRowRhs);
""")

COPT_CreateEnv   = coptlib.COPT_CreateEnv
COPT_DeleteEnv   = coptlib.COPT_DeleteEnv

COPT_CreateProb  = coptlib.COPT_CreateProb
COPT_DeleteProb  = coptlib.COPT_DeleteProb

COPT_AddCol        = coptlib.COPT_AddCol
COPT_AddRow        = coptlib.COPT_AddRow
COPT_AddLazyConstr = coptlib.COPT_AddLazyConstr
COPT_AddUserCut    = coptlib.COPT_AddUserCut
COPT_AddSOSs       = coptlib.COPT_AddSOSs
COPT_GetCols       = coptlib.COPT_GetCols
COPT_GetRows       = coptlib.COPT_GetRows
COPT_DelCols       = coptlib.COPT_DelCols
COPT_DelRows       = coptlib.COPT_DelRows

COPT_SetObjSense = coptlib.COPT_SetObjSense
COPT_SetObjConst = coptlib.COPT_SetObjConst

COPT_SetColObj   = coptlib.COPT_SetColObj
COPT_SetColType  = coptlib.COPT_SetColType
COPT_SetColLower = coptlib.COPT_SetColLower
COPT_SetColUpper = coptlib.COPT_SetColUpper
COPT_SetColNames = coptlib.COPT_SetColNames

COPT_SetRowLower = coptlib.COPT_SetRowLower
COPT_SetRowUpper = coptlib.COPT_SetRowUpper
COPT_SetRowNames = coptlib.COPT_SetRowNames

COPT_ReplaceColObj = coptlib.COPT_ReplaceColObj

COPT_ReadMps     = coptlib.COPT_ReadMps
COPT_ReadLp      = coptlib.COPT_ReadLp
COPT_ReadCbf     = coptlib.COPT_ReadCbf
COPT_ReadBin     = coptlib.COPT_ReadBin
COPT_ReadSol     = coptlib.COPT_ReadSol
COPT_ReadBasis   = coptlib.COPT_ReadBasis
COPT_ReadMst     = coptlib.COPT_ReadMst
COPT_ReadParam   = coptlib.COPT_ReadParam

COPT_WriteMps    = coptlib.COPT_WriteMps
COPT_WriteLp     = coptlib.COPT_WriteLp
COPT_WriteCbf    = coptlib.COPT_WriteCbf
COPT_WriteBin    = coptlib.COPT_WriteBin
COPT_WriteSol    = coptlib.COPT_WriteSol
COPT_WriteBasis  = coptlib.COPT_WriteBasis
COPT_WriteMst    = coptlib.COPT_WriteMst
COPT_WriteParam  = coptlib.COPT_WriteParam

COPT_AddMipStart = coptlib.COPT_AddMipStart

COPT_Solve       = coptlib.COPT_Solve
COPT_SolveLp     = coptlib.COPT_SolveLp

COPT_GetPoolObjVal   = coptlib.COPT_GetPoolObjVal
COPT_GetPoolSolution = coptlib.COPT_GetPoolSolution

COPT_SetIntParam    = coptlib.COPT_SetIntParam
COPT_GetIntParam    = coptlib.COPT_GetIntParam
COPT_GetIntParamDef = coptlib.COPT_GetIntParamDef

COPT_SetDblParam    = coptlib.COPT_SetDblParam
COPT_GetDblParam    = coptlib.COPT_GetDblParam
COPT_GetDblParamDef = coptlib.COPT_GetDblParamDef

COPT_Reset = coptlib.COPT_Reset

COPT_GetIntAttr  = coptlib.COPT_GetIntAttr
COPT_GetDblAttr  = coptlib.COPT_GetDblAttr

COPT_GetColIdx   = coptlib.COPT_GetColIdx
COPT_GetRowIdx   = coptlib.COPT_GetRowIdx

COPT_GetColInfo  = coptlib.COPT_GetColInfo
COPT_GetRowInfo  = coptlib.COPT_GetRowInfo

COPT_GetColType  = coptlib.COPT_GetColType
COPT_GetColBasis = coptlib.COPT_GetColBasis
COPT_GetRowBasis = coptlib.COPT_GetRowBasis

COPT_GetColName  = coptlib.COPT_GetColName
COPT_GetRowName  = coptlib.COPT_GetRowName

COPT_SetCallback           = coptlib.COPT_SetCallback
COPT_GetCallbackInfo       = coptlib.COPT_GetCallbackInfo
COPT_AddCallbackSolution   = coptlib.COPT_AddCallbackSolution
COPT_AddCallbackUserCut    = coptlib.COPT_AddCallbackUserCut
COPT_AddCallbackLazyConstr = coptlib.COPT_AddCallbackLazyConstr

COPT_CBCONTEXT_MIPRELAX   = 0x1
COPT_CBCONTEXT_MIPSOL     = 0x2

COPT_CBINFO_BESTOBJ       = "BestObj"
COPT_CBINFO_BESTBND       = "BestBnd"
COPT_CBINFO_HASINCUMBENT  = "HasIncumbent"
COPT_CBINFO_INCUMBENT     = "Incumbent"
COPT_CBINFO_MIPCANDIDATE  = "MipCandidate"
COPT_CBINFO_MIPCANDOBJ    = "MipCandObj"
COPT_CBINFO_RELAXSOLUTION = "RelaxSolution"
COPT_CBINFO_RELAXSOLOBJ   = "RelaxSolObj"

# Optimation status map
coptstat    = {0:  OptimizationStatus.LOADED,     # unstarted
               1:  OptimizationStatus.OPTIMAL,    # optimal
               2:  OptimizationStatus.INFEASIBLE, # infeasible
               3:  OptimizationStatus.UNBOUNDED,  # unbounded
               4:  OptimizationStatus.INFEASIBLE, # inf_or_unbd
               5:  OptimizationStatus.OTHER,      # numerical
               6:  OptimizationStatus.OTHER,      # node limit
               7:  OptimizationStatus.FEASIBLE,   # imprecise
               8:  OptimizationStatus.OTHER,      # time out
               9:  OptimizationStatus.OTHER,      # unfinished
               10: OptimizationStatus.OTHER}      # interrupted

# Constraint senses map
coptrsense  = {EQUAL:            'E',
               LESS_OR_EQUAL:    'L',
               GREATER_OR_EQUAL: 'G'}

class SolverCopt(Solver):
  def __init__(self, model: Model, name: str="", sense: str="", modelp: CData=ffi.NULL):
    super().__init__(model, name, sense)

    # Initalize data
    self._ownsmodel = True
    self._modelname = name

    self._pcoptenv  = ffi.NULL
    self._pcoptprob = ffi.NULL
    self._coptenv   = ffi.NULL
    self._coptprob  = ffi.NULL

    self._callback  = None
    self._nlazy     = 0
    self._nusercut  = 0

    # Create COPT environment
    if modelp == ffi.NULL:
      self._ownsmodel = True
      self._pcoptenv = ffi.new("copt_env **")
      rc = COPT_CreateEnv(self._pcoptenv)
      if rc != 0:
        raise InterfacingError("Failed to create COPT environment")
      self._coptenv = self._pcoptenv[0]

      # Create COPT problem
      self._pcoptprob = ffi.new("copt_prob **")
      rc = COPT_CreateProb(self._coptenv, self._pcoptprob)
      if rc != 0:
        raise InterfacingError("Failed to create COPT model")
      self._coptprob = self._pcoptprob[0]
    else:
      self._ownsmodel = False
      self._coptprob = modelp

    # Set objective sense
    if sense:
      self.set_objective_sense(sense)

    # Default number of threads
    self.__threads = 0

    # Where solution will be stored
    self.__x = EmptyVarSol(model)
    self.__rc = EmptyVarSol(model)
    self.__pi = EmptyRowSol(model)
    self.__obj_val = None

  def __clear_sol(self):
    model = self.model
    self.__x = EmptyVarSol(model)
    self.__rc = EmptyVarSol(model)
    self.__pi = EmptyRowSol(model)
    self.__obj_val = None

  def __del__(self):
    if self._ownsmodel:
      if self._coptprob:
        COPT_DeleteProb(self._pcoptprob)
      if self._coptenv:
        COPT_DeleteEnv(self._pcoptenv)

  def add_var(self,
              obj: float = 0,
              lb: float = 0,
              ub: float = INF,
              var_type: str = CONTINUOUS,
              column: "Column"=None,
              name: str=""):
    nz = 0 if column is None else len(column.constrs)

    if nz:
      vind = ffi.new("int[]", [c.idx for c in column.constrs])
      vval = ffi.new("double[]", [column.coeffs[i] for i in range(nz)])
    else:
      vind = ffi.NULL
      vval = ffi.NULL

    vtype = var_type.encode('utf-8')

    if not name:
      name = 'c({})'.format(self.num_cols())

    rc = COPT_AddCol(self._coptprob, obj, nz, vind, vval, vtype,
                     lb, ub, name.encode('utf-8'))
    if rc != 0:
      raise ParameterNotAvailable("Failed to add variable '{}' to model.".format(name))

  def add_constr(self, lin_expr: "LinExpr", name: str=""):
    nz = len(lin_expr.expr)
    cind = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
    cval = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])

    sense = coptrsense[lin_expr.sense].encode('utf-8')
    rhs   = -lin_expr.const

    if not name:
      name = 'r({})'.format(self.num_rows())

    rc = COPT_AddRow(self._coptprob, nz, cind, cval, sense, rhs, 0, name.encode('utf-8'))
    if rc != 0:
      raise ParameterNotAvailable("Failed to add constraint '{}' to model".format(name))

  def add_sos(self, sos: List[Tuple["Var", float]], sos_type: int):
    types = ffi.new("int[]", [sos_type])
    rbeg  = ffi.new("int[]", [0, len(sos)])
    rind  = ffi.new("int[]", [v.idx for (v, f) in sos])
    wval  = ffi.new("double[]", [f for (v, f) in sos])

    rc = COPT_AddSOSs(self._coptprob, 1, types, rbeg, ffi.NULL, rind, wval)
    if rc != 0:
      raise ParameterNotAvailable("Failed to add SOS constraint to model")

  def add_cut(self, lin_expr: "LinExpr"):
    nz = len(lin_expr.expr)
    cind = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
    cval = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])

    sense = coptrsense[lin_expr.sense].encode('utf-8')
    rhs   = -lin_expr.const

    name = 'usrcut({})'.format(self._nusercut)

    rc = COPT_AddUserCut(self._coptprob, nz, cind, cval, sense, rhs, 0, name.encode('utf-8'))
    if rc != 0:
      raise ParameterNotAvailable("Failed to add user cut '{}' to model".format(name))

    self._nusercut += 1

  def add_lazy_constr(self, lin_expr: "LinExpr"):
    nz = len(lin_expr.expr)
    cind = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
    cval = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])

    sense = coptrsense[lin_expr.sense].encode('utf-8')
    rhs   = -lin_expr.const

    name = 'lz({})'.format(self._nlazy)

    rc = COPT_AddLazyConstr(self._coptprob, nz, cind, cval, sense, rhs, 0, name.encode('utf-8'))
    if rc != 0:
      raise ParameterNotAvailable("Failed to add lazy constraint '{}' to model".format(name))

    self._nlazy += 1

  def relax(self):
    idxv = [var.idx for var in self.model.vars if var.var_type in [BINARY, INTEGER]]

    n = len(idxv)
    idxs = ffi.new("int[]", idxv)

    cont_char = 'C'.encode('utf-8')
    ccont = ffi.new("char[]", [cont_char for i in range(n)])

    if idxv:
      rc = COPT_SetColType(self._coptprob, n, idxs, ccont)
      if rc != 0:
        raise ParameterNotAvailable("Failed to relax model")

  def optimize(self, relax: bool = False) -> OptimizationStatus:
    @ffi.callback(
      """
      int (copt_prob *, void *, int, void *)
      """
    )
    def callback(
      p_model: CData, p_cbdata: CData, where: int, p_usrdata: CData
    ) -> int:
      if self.model.cuts_generator and where == COPT_CBCONTEXT_MIPRELAX:
        mcc = ModelCoptCB(p_model, p_cbdata, where)
        self.model.cuts_generator.generate_constrs(mcc)
        return 0
      
      if self.model.lazy_constrs_generator and where == COPT_CBCONTEXT_MIPSOL:
        mcc = ModelCoptCB(p_model, p_cbdata, where)
        self.model.lazy_constrs_generator.generate_constrs(mcc)
        return 0
      
      return 0
    
    if (self.model.cuts_generator is not None or
        self.model.lazy_constrs_generator is not None):
      COPT_SetCallback(self._coptprob, callback, COPT_CBCONTEXT_MIPRELAX | COPT_CBCONTEXT_MIPSOL, ffi.NULL)

    if self.__threads >= 1:
      self.set_int_param("Threads", self.__threads)

    self.set_int_param("CutLevel", self.model.cuts)
    self.set_int_param("Presolve", self.model.preprocess)

    self.set_dbl_param("IntTol", self.model.integer_tol)
    self.set_dbl_param("FeasTol", self.model.infeas_tol)
    self.set_dbl_param("DualTol", self.model.opt_tol)
    
    if self.model.lp_method == LP_Method.BARRIER:
      self.set_int_param("LpMethod", 2)
    elif self.model.lp_method == LP_Method.DUAL:
      self.set_int_param("LpMethod", 1)

    self.set_mip_gap(self.model.max_mip_gap)

    # Clear old solution
    self.__clear_sol()

    if relax:
      rc = COPT_SolveLp(self._coptprob)
    else:
      rc = COPT_Solve(self._coptprob)
    if rc != 0:
      raise InterfacingError("Failed to solve model")

    ismip = self.get_int_attr("IsMIP")
    if ismip and not relax:
      mipstat  = self.get_int_attr("MipStatus")
      probstat = coptstat[mipstat]

      if mipstat in [1, 6, 8, 10]: # Optimal, Timeout, Nodelimit, Interrupted
        nsols = self.get_num_solutions()
        if nsols >= 1:
          self.__obj_val = self.get_dbl_attr("BestObj")
          self.__x = ffi.new("double[{}]".format(self.num_cols()))

          rc = COPT_GetColInfo(self._coptprob, "Value".encode('utf-8'), self.num_cols(), ffi.NULL, self.__x)
          if rc != 0:
            raise ParameterNotAvailable("Failed to get solution")

          if mipstat == 1:
            return OptimizationStatus.OPTIMAL
          else:
            return OptimizationStatus.FEASIBLE
        else:
          return OptimizationStatus.NO_SOLUTION_FOUND
    else:
      lpstat = self.get_int_attr("LpStatus")
      probstat = coptstat[lpstat]

      if lpstat == 1: # Optimal
        if isinstance(self.__x, EmptyVarSol):
          self.__obj_val = self.get_dbl_attr("LpObjVal")
          self.__x = ffi.new("double[{}]".format(self.num_cols()))
          self.__pi = ffi.new("double[{}]".format(self.num_rows()))
          self.__rc = ffi.new("double[{}]".format(self.num_cols()))

          rc = COPT_GetColInfo(self._coptprob, "Value".encode('utf-8'), self.num_cols(), ffi.NULL, self.__x)
          if rc != 0:
            raise ParameterNotAvailable("Failed to get solution")

          rc = COPT_GetColInfo(self._coptprob, "RedCost".encode('utf-8'), self.num_cols(), ffi.NULL, self.__rc)
          if rc != 0:
            raise ParameterNotAvailable("Failed to get reduced costs")

          rc = COPT_GetRowInfo(self._coptprob, "Dual".encode('utf-8'), self.num_rows(), ffi.NULL, self.__pi)
          if rc != 0:
            raise ParameterNotAvailable("Failed to get shadow price")

    return probstat

  def read(self, file_path: str):
    if not os.path.isfile(file_path):
      raise FileNotFoundError("File '{}' does not exists".format(file_path))

    file_name, file_ext = os.path.splitext(file_path)
    if file_ext == '.gz':
      file_name, file_ext = os.path.splitext(file_name)

    if not file_ext:
      raise ValueError("Failed to determine input file type")
    elif file_ext == '.mps':
      rc = COPT_ReadMps(self._coptprob, file_path.encode('utf-8'))
    elif file_ext == '.lp':
      rc = COPT_ReadLp(self._coptprob, file_path.encode('utf-8'))
    elif file_ext == '.cbf':
      rc = COPT_ReadCbf(self._coptprob, file_path.encode('utf-8'))
    elif file_ext == '.bin':
      rc = COPT_ReadBin(self._coptprob, file_path.encode('utf-8'))
    elif file_ext == '.sol':
      rc = COPT_ReadSol(self._coptprob, file_path.encode('utf-8'))
    elif file_ext == '.bas':
      rc = COPT_ReadBasis(self._coptprob, file_path.encode('utf-8'))
    elif file_ext == '.mst':
      rc = COPT_ReadMst(self._coptprob, file_path.encode('utf-8'))
    elif file_ext == '.par':
      rc = COPT_ReadParam(self._coptprob, file_path.encode('utf-8'))
    else:
      raise ValueError("Unsupported file type")

    if rc != 0:
      raise InterfacingError("Failed to read file '{}'".format(file_path))

  def write(self, file_path: str):
    file_name, file_ext = os.path.splitext(file_path)

    if not file_ext:
      raise ValueError("Failed to determine output file type")
    elif file_ext == '.mps':
      rc = COPT_WriteMps(self._coptprob, file_path.encode('utf-8'))
    elif file_ext == '.lp':
      rc = COPT_WriteLp(self._coptprob, file_path.encode('utf-8'))
    elif file_ext == '.cbf':
      rc = COPT_WriteCbf(self._coptprob, file_path.encode('utf-8'))
    elif file_ext == '.bin':
      rc = COPT_WriteBin(self._coptprob, file_path.encode('utf-8'))
    elif file_ext == '.sol':
      rc = COPT_WriteSol(self._coptprob, file_path.encode('utf-8'))
    elif file_ext == '.bas':
      rc = COPT_WriteBasis(self._coptprob, file_path.encode('utf-8'))
    elif file_ext == '.mst':
      rc = COPT_WriteMst(self._coptprob, file_path.encode('utf-8'))
    elif file_ext == '.par':
      rc = COPT_WriteParam(self._coptprob, file_path.encode('utf-8'))
    else:
      raise ValueError("Unsupported file type")

    if rc != 0:
      raise InterfacingError("Failed to write file '{}'".format(file_path))

  def num_cols(self) -> int:
    return self.get_int_attr("Cols")
  
  def num_int(self) -> int:
    return self.get_int_attr("Ints") + self.get_int_attr("Bins")

  def num_rows(self) -> int:
    return self.get_int_attr("Rows")

  def num_nz(self) -> int:
    return self.get_int_attr("Elems")

  def get_problem_name(self) -> str:
    return self._modelname

  def set_problem_name(self, name: str):
    self._modelname = name

  def get_objective(self) -> "LinExpr":
    obj = ffi.new("double[]", [0.0 for i in range(self.num_cols())])

    rc = COPT_GetColInfo(self._coptprob, "Obj".encode("utf-8"), self.num_cols(), ffi.NULL, obj)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get objective cost")
    
    obj_expr = xsum(
      obj[i] * self.models.vars[i]
      for i in range(self.num_cols())
      if abs(obj[i]) > 1e-10
    )
    obj_expr.add_const(self.get_objective_const())
    obj_expr.sense = self.get_objective_sense()
    return obj_expr

  def set_objective(self, lin_expr: "LinExpr", sense: str = ""):
    nz = len(lin_expr.expr)
    cind = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
    cval = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])
    const = lin_expr.const

    rc = COPT_ReplaceColObj(self._coptprob, nz, cind, cval)
    if rc != 0:
      raise ParameterNotAvailable("Failed to set objective")

    rc = COPT_SetObjConst(self._coptprob, const)
    if rc != 0:
      raise ParameterNotAvailable("Failed to set objective constant")

    # Default to minimization
    if MAXIMIZE in (lin_expr.sense, sense):
      rc = COPT_SetObjSense(self._coptprob, -1)
    elif MINIMIZE in (lin_expr.sense, sense):
      rc = COPT_SetObjSense(self._coptprob, 1)

    if rc != 0:
      raise ParameterNotAvailable("Failed to set objective sense")

  def get_objective_sense(self) -> str:
    isense = self.get_int_attr("ObjSense")

    if isense == 1:
      return MINIMIZE
    elif isense == -1:
      return MAXIMIZE
    else:
      raise ValueError("Unknown objective sense")

  def set_objective_sense(self, sense: str):
    if sense.strip().upper() == MAXIMIZE.strip().upper():
      rc = COPT_SetObjSense(self._coptprob, -1)
    elif sense.strip().upper() == MINIMIZE.strip().upper():
      rc = COPT_SetObjSense(self._coptprob, 1)
    else:
      raise ValueError("Unknown objective sense '{}'".format(sense))

    if rc != 0:
      raise ParameterNotAvailable("Failed to set objective sense")

  def get_objective_value(self) -> float:
    return self.__obj_val

  def get_objective_value_i(self, i: int) -> float:
    objval_i = ffi.new("double *")

    rc = COPT_GetPoolObjVal(self._coptprob, i, objval_i)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get i-th pool objective value")
    
    return objval_i[0]

  def get_objective_bound(self) -> float:
    return self.get_dbl_attr("BestBnd")

  def get_objective_const(self) -> float:
    return self.get_dbl_attr("ObjConst")

  def set_objective_const(self, const: float):
    rc = COPT_SetObjConst(self._coptprob, const)
    if rc != 0:
      raise ParameterNotAvailable("Failed to set objective constant")

  def get_cutoff(self) -> float:
    print("Cutoff not supported yet, just return infinity")
    return INF

  def set_cutoff(self, cutoff: float):
    print("Cutoff not supported yet, just ignore it")

  def get_mip_gap(self) -> float:
    return self.get_dbl_attr("BestGap")

  def set_mip_gap(self, mip_gap: float):
    self.set_dbl_param("RelGap", mip_gap)

  def get_mip_gap_abs(self) -> float:
    bestobj = self.get_dbl_attr("BestObj")
    bestbnd = self.get_dbl_attr("BestBnd")
    return math.fabs(bestobj - bestbnd)

  def set_mip_gap_abs(self, allowable_gap: float):
    self.set_dbl_param("AbsGap", allowable_gap)

  def get_max_seconds(self) -> float:
    return self.get_dbl_param("TimeLimit")

  def set_max_seconds(self, max_seconds: float):
    self.set_dbl_param("TimeLimit", max_seconds)

  def get_max_solutions(self) -> int:
    print("Get max solutions not supported yet, just ignore it")

  def set_max_solutions(self, max_solutions: int):
    print("Set max solutions not supported yet, just ignore it")

  def get_max_nodes(self) -> int:
    return self.get_int_attr("NodeCnt")

  def set_max_nodes(self, max_nodes: int):
    self.set_int_param("NodeLimit", max_nodes)

  def set_num_threads(self, threads: int):
    self.__threads = threads

  def set_pump_passes(self, passes: int):
    print("FeasPump passes not supported yet, just ignore it")

  def get_pump_passes(self) -> int:
    print("FeasPump passes not supported yet, just return -1")
    return -1

  def get_num_solutions(self) -> int:
    ismip = self.get_int_attr("IsMip")
    if ismip:
      npoolsol = self.get_int_attr("PoolSols")
      if npoolsol > 0:
        return npoolsol
      else:
        return self.get_int_attr("HasMipSol")
    else:
      return self.get_int_attr("HasLpSol")

  def get_log(self) -> List[Tuple[float, Tuple[float, float]]]:
    raise NotImplementedError("Unsupported function of COPT")

  def get_verbose(self) -> int:
    return self.get_int_param("Logging")

  def set_verbose(self, verbose: int):
    return self.set_int_param("Logging", verbose)

  def set_processing_limits(self,
                            max_time: numbers.Real = mip.INF,
                            max_nodes: int = mip.INT_MAX,
                            max_sol: int = mip.INT_MAX,
                            max_seconds_same_incumbent: float = mip.INF,
                            max_nodes_same_incumbent: int = mip.INT_MAX):
    if max_time != mip.INF:
      self.set_dbl_param("TimeLimit", max_time)
    if max_nodes != mip.INT_MAX:
      self.set_int_param("NodeLimit", max_nodes)

  def get_emphasis(self) -> SearchEmphasis:
    print("MIP emphasis not supported yet, just return optimality")
    return SearchEmphasis.OPTIMALITY

  def set_emphasis(self, emph: SearchEmphasis):
    print("MIP emphasis not supported yet, just ignore it")

  def set_start(self, start: List[Tuple["Var", float]]):
    nz = len(start)
    cind = ffi.new("int[]", [v.idx for (v, f) in start])
    cval = ffi.new("double[]", [float(f) for (v, f) in start])

    rc = COPT_AddMipStart(self._coptprob, nz, cind, cval)
    if rc != 0:
      raise ParameterNotAvailable("Failed to add MIP start information")

  def var_get_lb(self, var: "Var") -> float:
    return self.get_dbl_colinfo("LB", var.idx)

  def var_set_lb(self, var: "Var", value: float):
    self.set_dbl_colinfo("LB", var.idx, value)

  def var_get_ub(self, var: "Var") -> float:
    return self.get_dbl_colinfo("UB", var.idx)

  def var_set_ub(self, var: "Var", value: float):
    self.set_dbl_colinfo("UB", var.idx, value)

  def var_get_obj(self, var: "Var") -> float:
    return self.get_dbl_colinfo("Obj", var.idx)

  def var_set_obj(self, var: "Var", value: float):
    self.set_dbl_colinfo("Obj", var.idx, value)

  def var_get_var_type(self, var: "Var") -> str:
    vtype = ffi.new("char *")
    vidx = ffi.new("int *", var.idx)

    rc = COPT_GetColType(self._coptprob, 1, vidx, vtype)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get type of variable {}".format(var.idx))

    return vtype[0].decode('utf-8')

  def var_set_var_type(self, var: "Var", value: str):
    idx = ffi.new("int *", var.idx)

    rc = COPT_SetColType(self._coptprob, 1, idx, value.encode('utf-8'))
    if rc != 0:
      raise ParameterNotAvailable("Failed to set type for variable {}".format(var.idx))

  def var_get_column(self, var: "Var") -> "Column":
    nnz = ffi.new("int*")
    idx = ffi.new("int*", var.idx)
    
    rc = COPT_GetCols(self._coptprob, 1, idx, ffi.NULL, ffi.NULL,
                      ffi.NULL, ffi.NULL, 0, nnz)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get info of variable {}".format(var.idx))

    cbeg = ffi.new("int[2]")
    cind = ffi.new("int[{}]".format(nnz[0]))
    cval = ffi.new("double[{}]".format(nnz[0]))

    rc = COPT_GetCols(self._coptprob, 1, idx, cbeg, ffi.NULL, cind, cval,
                      nnz[0], ffi.NULL)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get contents of variable {}".format(var.idx))

    constr = [self.model.constrs[cind[i]] for i in range(nnz[0])]
    coefs = [float(cval[i]) for i in range(nnz[0])]

    return Column(constr, coefs)

  def var_set_column(self, var: "Var", value: "Column"):
    raise NotImplementedError("Unsupported function of COPT")

  def var_get_x(self, var: "Var") -> float:
    return self.__x[var.idx]

  def var_get_xi(self, var: "Var", i: int) -> float:
    idx = ffi.new("int *", var.idx)
    var_xi = ffi.new("double *")

    rc = COPT_GetPoolSolution(self._coptprob, i, 1, idx, var_xi)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get i-th pool solution of variable '{}'".format(var.name))
    
    return var_xi[0]

  def var_get_rc(self, var: "Var") -> float:
    return self.__rc[var.idx]

  def var_get_name(self, idx: int) -> str:
    cname = ffi.new("char[{}]".format(COPT_BUFFSIZE))

    rc = COPT_GetColName(self._coptprob, idx, cname, COPT_BUFFSIZE, ffi.NULL)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get name of variable {}".format(idx))
    
    return ffi.string(cname).decode('utf-8')

  def var_get_index(self, name: str) -> int:
    idx = ffi.new("int *")

    rc = COPT_GetColIdx(self._coptprob, name.encode('utf-8'), idx)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get index of variable '{}'".format(name))

    return idx[0]

  def remove_vars(self, varsList: List[int]):
    idx = ffi.new("int[]", varsList)

    rc = COPT_DelCols(self._coptprob, len(varsList), idx)
    if rc != 0:
      raise ParameterNotAvailable("Failed to delete variables")

  def constr_get_expr(self, constr: "Constr") -> "LinExpr":
    nnz = ffi.new("int *")
    idx = ffi.new("int *", constr.idx)

    rc = COPT_GetRows(self._coptprob, 1, idx, ffi.NULL, ffi.NULL,
                      ffi.NULL, ffi.NULL, 0, nnz)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get info of constraint {}".format(constr.idx))
    
    rbeg = ffi.new("int[2]")
    rind = ffi.new("int[{}]".format(nnz[0]))
    rval = ffi.new("double[{}]".format(nnz[0]))

    rc = COPT_GetRows(self._coptprob, 1, idx, rbeg, ffi.NULL, rind, rval,
                      nnz[0], ffi.NULL)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get contents of constraint {}".format(constr.idx))

    rsense = self.constr_get_sense(idx[0])
    rhs = self.constr_get_rhs(idx[0])

    expr = LinExpr(const=-rhs, sense=rsense)
    for i in range(nnz[0]):
      expr.add_var(self.model.vars[rind[i]], rval[i])
    
    return expr

  def constr_set_expr(self, constr: "Constr", value: "LinExpr") -> "LinExpr":
    raise NotImplementedError("Unsupported functionality of COPT")

  def constr_get_sense(self, idx: int) -> str:
    #TODO: No rowsense in COPT
    lb = self.get_dbl_rowinfo("LB", idx)
    ub = self.get_dbl_rowinfo("UB", idx)
    cinf = self.get_dbl_param("InfBound")

    if lb <= -cinf:
      return LESS_OR_EQUAL
    elif ub >= +cinf:
      return GREATER_OR_EQUAL
    else:
      return EQUAL

  def constr_get_rhs(self, idx: int) -> float:
    #TODO: No rhs in COPT
    rsense = self.constr_get_sense(idx)

    if rsense == LESS_OR_EQUAL:
      return self.get_dbl_rowinfo("UB", idx)
    elif rsense == GREATER_OR_EQUAL:
      return self.get_dbl_rowinfo("LB", idx)
    else:
      return self.get_dbl_rowinfo("LB", idx)

  def constr_set_rhs(self, idx: int, rhs: float):
    #TODO: No rhs in COPT
    rsense = self.constr_get_sense(idx)

    if rsense == LESS_OR_EQUAL:
      self.set_dbl_rowinfo("UB", idx, rhs)
    elif rsense == GREATER_OR_EQUAL:
      self.set_dbl_rowinfo("LB", idx, rhs)
    else:
      self.set_dbl_rowinfo("LB", idx, rhs)

  def constr_get_slack(self, constr: "Constr") -> float:
    return self.get_dbl_rowinfo("Slack", constr.idx)

  def constr_get_pi(self, constr: "Constr") -> float:
    return self.__pi[constr.idx]

  def constr_get_name(self, idx: int) -> str:
    rname = ffi.new("char[{}]".format(COPT_BUFFSIZE))

    rc = COPT_GetRowName(self._coptprob, idx, rname, COPT_BUFFSIZE, ffi.NULL)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get name of constraint {}".format(idx))
    
    return ffi.string(rname).decode('utf-8')

  def constr_get_index(self, name: str) -> int:
    idx = ffi.new("int *")

    rc = COPT_GetRowIdx(self._coptprob, name.encode('utf-8'), idx)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get index of constraint '{}'".format(name))

    return idx[0]

  def remove_constrs(self, constrsList: List[int]):
    idx = ffi.new("int[]", constrsList)

    rc = COPT_DelRows(self._coptprob, len(constrsList), idx)
    if rc != 0:
      raise ParameterNotAvailable("Failed to delete constraints")

  def reset(self):
    self.__x = EmptyVarSol(self.model)
    self.__rc = EmptyVarSol(self.model)
    self.__pi = EmptyRowSol(self.model)
    self.__obj_val = None
    rc = COPT_Reset(self._coptprob, 0)
    if rc != 0:
      raise ParameterNotAvailable("Failed to reset COPT model")

  def get_int_attr(self, name: str) -> int:
    int_attr = ffi.new("int *")

    rc = COPT_GetIntAttr(self._coptprob, name.encode('utf-8'), int_attr)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get integer attribute '{}'".format(name))

    return int_attr[0]

  def get_dbl_attr(self, name: str) -> float:
    dbl_attr = ffi.new("double *")

    rc = COPT_GetDblAttr(self._coptprob, name.encode('utf-8'), dbl_attr)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get double attribute '{}'".format(name))

    return dbl_attr[0]

  def get_dbl_colinfo(self, name: str, index: int) -> float:
    idx = ffi.new("int *", index)
    dbl_info = ffi.new("double *")

    rc = COPT_GetColInfo(self._coptprob, name.encode('utf-8'), 1, idx, dbl_info)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get double variable info '{}'".format(name))

    return dbl_info[0]

  def set_dbl_colinfo(self, name: str, index: int, value: float):
    idx = ffi.new("int *", index)
    dbl_info = ffi.new("double *", value)

    if name == "Obj":
      rc = COPT_SetColObj(self._coptprob, 1, idx, dbl_info)
    elif name == "LB":
      rc = COPT_SetColLower(self._coptprob, 1, idx, dbl_info)
    elif name == "UB":
      rc = COPT_SetColUpper(self._coptprob, 1, idx, dbl_info)
    else:
      raise ParameterNotAvailable("Unknown variable info '{}'".format(name))

    if rc != 0:
      raise ParameterNotAvailable("Failed to set double variable info '{}'".format(name))

  def get_dbl_rowinfo(self, name: str, index: int) -> float:
    idx = ffi.new("int *", index)
    dbl_info = ffi.new("double *")

    rc = COPT_GetRowInfo(self._coptprob, name.encode('utf-8'), 1, idx, dbl_info)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get double constraint info '{}'".format(name))

    return dbl_info[0]

  def set_dbl_rowinfo(self, name: str, index: int, value: float):
    idx = ffi.new("int *", index)
    dbl_info = ffi.new("double *", value)

    if name == "LB":
      rc = COPT_SetRowLower(self._coptprob, 1, idx, dbl_info)
    elif name == "UB":
      rc = COPT_SetRowUpper(self._coptprob, 1, idx, dbl_info)
    else:
      raise ParameterNotAvailable("Unknown constraint info '{}'".format(name))

    if rc != 0:
      raise ParameterNotAvailable("Failed to set double constraint info '{}'".format(name))

  def get_int_param_default(self, name: str) -> int:
    int_param = ffi.new("int *")

    rc = COPT_GetIntParamDef(self._coptprob, name.encode('utf-8'), int_param)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get integer parameter '{}'".format(name))

    return int_param[0]

  def get_int_param(self, name: str) -> int:
    int_param = ffi.new("int *")

    rc = COPT_GetIntParam(self._coptprob, name.encode('utf-8'), int_param)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get integer parameter '{}'".format(name))

    return int_param[0]

  def set_int_param(self, name: str, value: int):
    if value != self.get_int_param_default(name):
      rc = COPT_SetIntParam(self._coptprob, name.encode('utf-8'), value)
      if rc != 0:
        raise ParameterNotAvailable("Failed to set integer parameter '{0}' to value {1}".format(name, value))

  def get_dbl_param_default(self, name: str) -> float:
    dbl_param = ffi.new("double *")

    rc = COPT_GetDblParamDef(self._coptprob, name.encode('utf-8'), dbl_param)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get double parameter '{}'".format(name))

    return dbl_param[0]

  def get_dbl_param(self, name: str) -> float:
    dbl_param = ffi.new("double *")

    rc = COPT_GetDblParam(self._coptprob, name.encode('utf-8'), dbl_param)
    if rc != 0:
      raise ParameterNotAvailable("Failed to get double parameter '{}'".format(name))

    return dbl_param[0]

  def set_dbl_param(self, name: str, value: float):
    if value != self.get_dbl_param_default(name):
      rc = COPT_SetDblParam(self._coptprob, name.encode('utf-8'), value)
      if rc != 0:
        raise ParameterNotAvailable("Failed to set double parameter '{0}' to value {1}".format(name, value))

class SolverCoptCB(SolverCopt):
  def __init__(self, 
               model: Model, 
               copt_model: CData = ffi.NULL, 
               cb_data: CData = ffi.NULL,
               where: int = -1):
    assert copt_model != ffi.NULL
    assert cb_data != ffi.NULL

    super().__init__(model, "", "", copt_model)

    self._cb_data = cb_data
    self._coptprob = copt_model
    self._where = where

    self._status = OptimizationStatus.LOADED
    self._cb_sol = None
    self._cb_objval = INF

    if where not in [COPT_CBCONTEXT_MIPRELAX, COPT_CBCONTEXT_MIPSOL]:
      return

    if where == COPT_CBCONTEXT_MIPSOL:
      self._status = OptimizationStatus.FEASIBLE

      pncols = ffi.new("int *")
      rc = COPT_GetIntAttr(copt_model, "Cols".encode("utf-8"), pncols)
      if rc != 0:
        raise ParameterNotAvailable("Failed to get number of variables in COPT callback")

      ncols = pncols[0]
      self._cb_sol = ffi.new("double[{}]".format(ncols))

      rc = COPT_GetCallbackInfo(cb_data, COPT_CBINFO_MIPCANDIDATE.encode("utf-8"), self._cb_sol)
      if rc != 0:
        raise ParameterNotAvailable("Failed to get candidate MIP solution in COPT callback")

      pobjval = ffi.new("double *")
      rc = COPT_GetCallbackInfo(cb_data, COPT_CBINFO_MIPCANDOBJ.encode("utf-8"), pobjval)
      if rc != 0:
        raise ParameterNotAvailable("Failed to get MIP candidate objective value in COPT callback")

      self._cb_objval = pobjval[0]
    elif where == COPT_CBCONTEXT_MIPRELAX:
      self._status = OptimizationStatus.OPTIMAL

      pncols = ffi.new("int *")
      rc = COPT_GetIntAttr(copt_model, "Cols".encode("utf-8"), pncols)
      if rc != 0:
        raise ParameterNotAvailable("Failed to get number of variables in COPT callback")

      ncols = pncols[0]
      self._cb_sol = ffi.new("double[{}]".format(ncols))

      rc = COPT_GetCallbackInfo(cb_data, COPT_CBINFO_RELAXSOLUTION.encode("utf-8"), self._cb_sol)
      if rc != 0:
        raise ParameterNotAvailable("Failed to get relaxation solution in COPT callback")
    else:
      self._cb_sol = ffi.NULL

  def add_cut(self, lin_expr: "LinExpr"):
    numnz = len(lin_expr.expr)

    cind = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
    cval = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])

    sense = coptrsense[lin_expr.sense].encode('utf-8')
    rhs = -lin_expr.const

    if self._where == COPT_CBCONTEXT_MIPRELAX:
      rc = COPT_AddCallbackUserCut(self._cb_data, numnz, cind, cval, sense, rhs)
      if rc != 0:
        raise ParameterNotAvailable("Failed to add user cut in COPT callback")
    elif self._where == COPT_CBCONTEXT_MIPSOL:
      rc = COPT_AddCallbackLazyConstr(self._cb_data, numnz, cind, cval, sense, rhs)
      if rc != 0:
        raise ParameterNotAvailable("Failed to add lazy constraint in COPT callback")

  def add_constr(self, lin_expr: "LinExpr", name: str=""):
    nz = len(lin_expr.expr)
    cind = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
    cval = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])

    sense = coptrsense[lin_expr.sense].encode('utf-8')
    rhs = -lin_expr.const

    if self._where == COPT_CBCONTEXT_MIPSOL:
      rc = COPT_AddCallbackLazyConstr(self._cb_data, nz, cind, cval, sense, rhs)
      if rc != 0:
        raise ParameterNotAvailable("Failed to add lazy constraint in COPT callback")
    elif self._where == COPT_CBCONTEXT_MIPRELAX:
      rc = COPT_AddCallbackUserCut(self._cb_data, nz, cind, cval, sense, rhs)
      if rc != 0:
        raise ParameterNotAvailable("Failed to add user cut in COPT callback")

  def add_lazy_constr(self, lin_expr: "LinExpr"):
    if self._where == COPT_CBCONTEXT_MIPSOL:
      nz = len(lin_expr.expr)
      cind = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
      cval = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])

      sense = coptrsense[lin_expr.sense].encode('utf-8')
      rhs = -lin_expr.const

      rc = COPT_AddCallbackLazyConstr(self._cb_data, nz, cind, cval, sense, rhs)
      if rc != 0:
        raise ParameterNotAvailable("Failed to add lazy constraint in COPT callback")
    elif self._where == COPT_CBCONTEXT_MIPRELAX:
      nz = len(lin_expr.expr)
      cind = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
      cval = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])

      sense = coptrsense[lin_expr.sense].encode('utf-8')
      rhs = -lin_expr.const

      rc = COPT_AddCallbackUserCut(self._cb_data, nz, cind, cval, sense, rhs)
      if rc != 0:
        raise ParameterNotAvailable("Failed to add user cut in COPT callback")
    else:
      raise ProgrammingError("Unsupported callback context in add_lazy_constr")

  def get_status(self):
    return self._status

  def var_get_x(self, var: Var):
    if self._cb_sol == ffi.NULL:
      raise SolutionNotAvailable("Solution is not available")

    return self._cb_sol[var.idx]

  def __del__(self):
    return

class ModelCoptCB(Model):
  def __init__(self,
               copt_model: CData = ffi.NULL,
               cb_data: CData = ffi.NULL,
               where: int = -1):
    self.solver_name = "coptcb"
    self.solver = SolverCoptCB(self, copt_model, cb_data, where)

    self.constrs = VConstrList(self)
    self.vars = VVarList(self)
    self.where = where

    self._status = self.solver.get_status()

    self.__constrs_generator = None
    self.__lazy_constrs_generator = None

  def add_constr(self, lin_expr: LinExpr, name: str="") -> "Constr":
    if self.where == COPT_CBCONTEXT_MIPRELAX:
      self.add_cut(lin_expr)
      return None
    
    self.add_lazy_constr(lin_expr)
    return None
