from mip.model import Model, Solver, Column, Var, LinExpr, Constr
from mip.constants import MAXIMIZE, MINIMIZE, CONTINUOUS, INTEGER, BINARY, \
    OptimizationStatus, EQUAL, LESS_OR_EQUAL, GREATER_OR_EQUAL, SearchEmphasis
from cffi import FFI
from ctypes.util import find_library
from sys import maxsize
from typing import List, Tuple
from os.path import isfile

try:
    found = False
    lib_path = None

    for major_ver in reversed(range(6, 10)):
        for minor_ver in reversed(range(0, 11)):
            lib_path = find_library('gurobi{}{}'.format(major_ver,
                                                        minor_ver))
            if lib_path is not None:
                break
        if lib_path is not None:
            break

    if lib_path is None:
        raise Exception("""Gurobi not found. Plase check if the
        Gurobi dynamic loadable library if reachable
        """)
    ffi = FFI()

    grblib = ffi.dlopen(lib_path)
    print('gurobi version {}.{} found'.format(major_ver,
                                              minor_ver))
    has_gurobi = True
except Exception:
    has_gurobi = False


if has_gurobi:
    CData = ffi.CData
    os_is_64_bit = maxsize > 2**32
    INF = float('inf')
    MAX_NAME_SIZE = 512  # for variables and constraints

    ffi.cdef("""
        typedef struct _GRBmodel GRBmodel;
        typedef struct _GRBenv GRBenv;

        typedef int(*gurobi_callback)(GRBmodel *model, void *cbdata,
                                      int where, void *usrdata);

        int GRBloadenv(GRBenv **envP, const char *logfilename);

        int GRBnewmodel(GRBenv *env, GRBmodel **modelP,
            const char *Pname, int numvars,
            double *obj, double *lb, double *ub, char *vtype,
            char **varnames);

        void GRBfreeenv(GRBenv *env);

        int GRBfreemodel(GRBmodel *model);

        int GRBgetintattr(GRBmodel *model, const char *attrname, int *valueP);

        int GRBsetintattr(GRBmodel *model, const char *attrname, int newvalue);

        int GRBgetintattrelement(GRBmodel *model, const char *attrname,
            int element, int *valueP);

        int GRBsetintattrelement(GRBmodel *model, const char *attrname,
            int element, int newvalue);

        int GRBgetdblattr(GRBmodel *model, const char *attrname,
            double *valueP);

        int GRBsetdblattr(GRBmodel *model, const char *attrname,
            double newvalue);

        int GRBgetdblattrarray(GRBmodel *model, const char *attrname,
            int first, int len, double *values);

        int GRBsetdblattrarray(GRBmodel *model, const char *attrname,
            int first, int len, double *newvalues);

        int GRBsetdblattrlist(GRBmodel *model, const char *attrname,
            int len, int *ind, double *newvalues);

        int GRBgetdblattrelement(GRBmodel *model, const char *attrname,
            int element, double *valueP);

        int GRBsetdblattrelement(GRBmodel *model, const char *attrname,
            int element, double newvalue);

        int GRBgetcharattrarray(GRBmodel *model, const char *attrname,
                      int first, int len, char *values);

        int GRBsetcharattrarray(GRBmodel *model, const char *attrname,
            int first, int len, char *newvalues);

        int GRBgetcharattrelement(GRBmodel *model, const char *attrname,
                                int element, char *valueP);
        int GRBsetcharattrelement(GRBmodel *model, const char *attrname,
                                int element, char newvalue);

        int GRBgetstrattrelement(GRBmodel *model, const char *attrname,
                            int element, char **valueP);

        int GRBgetintparam(GRBenv *env, const char *paramname, int *valueP);

        int GRBsetintparam(GRBenv *env, const char *paramname, int value);

        int GRBgetdblparam(GRBenv *env, const char *paramname, double *valueP);

        int GRBsetdblparam(GRBenv *env, const char *paramname, double value);

        int GRBsetobjectiven(GRBmodel *model, int index, 
                        int priority, double weight,
                        double abstol, double reltol, const char *name,
                        double constant, int lnz, int *lind, double *lval);

        int GRBaddvar(GRBmodel *model, int numnz, int *vind, double *vval,
                    double obj, double lb, double ub, char vtype,
                    const char *varname);

        int GRBaddconstr(GRBmodel *model, int numnz, int *cind, double *cval,
               char sense, double rhs, const char *constrname);

        int GRBgetconstrs(GRBmodel *model, int *numnzP, int *cbeg,
                int *cind, double *cval, int start, int len);

        int GRBgetvars(GRBmodel *model, int *numnzP, int *vbeg, int *vind,
             double *vval, int start, int len);

        int GRBgetvarbyname(GRBmodel *model, const char *name, int *indexP);

        int GRBgetconstrbyname(GRBmodel *model, const char *name, int *indexP);

        int GRBoptimize(GRBmodel *model);

        int GRBupdatemodel(GRBmodel *model);

        int GRBwrite(GRBmodel *model, const char *filename);

        int GRBreadmodel(GRBenv *env, const char *filename, GRBmodel **modelP);

        int GRBsetcharattrlist(GRBmodel *model, const char *attrname,
            int len, int *ind, char *newvalues);

        int GRBsetcallbackfunc(GRBmodel *model,
                     gurobi_callback grbcb,
                     void  *usrdata);

        int GRBcbget(void *cbdata, int where, int what, void *resultP);

        int GRBcbsetparam(void *cbdata, const char *paramname,
            const char *newvalue);

        int GRBcbsolution(void *cbdata, const double *solution,
            double *objvalP);

        int GRBcbcut(void *cbdata, int cutlen, const int *cutind,
            const double *cutval,
            char cutsense, double cutrhs);

        int GRBcblazy(void *cbdata, int lazylen, const int *lazyind,
            const double *lazyval, char lazysense, double lazyrhs);
    """)

    GRBloadenv = grblib.GRBloadenv
    GRBnewmodel = grblib.GRBnewmodel
    GRBfreeenv = grblib.GRBfreeenv
    GRBfreemodel = grblib.GRBfreemodel
    GRBaddvar = grblib.GRBaddvar
    GRBaddconstr = grblib.GRBaddconstr
    GRBoptimize = grblib.GRBoptimize
    GRBgetvarbyname = grblib.GRBgetvarbyname
    GRBsetdblattrarray = grblib.GRBsetdblattrarray
    GRBsetcharattrlist = grblib.GRBsetcharattrlist
    GRBsetdblattrlist = grblib.GRBsetdblattrlist
    GRBwrite = grblib.GRBwrite
    GRBreadmodel = grblib.GRBreadmodel
    GRBgetconstrbyname = grblib.GRBgetconstrbyname
    GRBupdatemodel = grblib.GRBupdatemodel
    GRBgetcharattrelement = grblib.GRBgetcharattrelement
    GRBgetconstrs = grblib.GRBgetconstrs
    GRBgetdblattrelement = grblib.GRBgetdblattrelement
    GRBgetvars = grblib.GRBgetvars
    GRBsetcharattrelement = grblib.GRBsetcharattrelement
    GRBsetdblattrelement = grblib.GRBsetdblattrelement
    GRBsetintattr = grblib.GRBsetintattr
    GRBsetdblattr = grblib.GRBsetdblattr
    GRBgetintattr = grblib.GRBgetintattr
    GRBgetintparam = grblib.GRBgetintparam
    GRBsetintparam = grblib.GRBsetintparam
    GRBgetdblattr = grblib.GRBgetdblattr
    GRBsetdblparam = grblib.GRBsetdblparam
    GRBgetdblparam = grblib.GRBgetdblparam
    GRBgetstrattrelement = grblib.GRBgetstrattrelement
    GRBcbget = grblib.GRBcbget
    GRBcbsetparam = grblib.GRBcbsetparam
    GRBcbsolution = grblib.GRBcbsolution
    GRBcbcut = grblib.GRBcbcut
    GRBcblazy = grblib.GRBcblazy
    GRBsetcallbackfunc = grblib.GRBsetcallbackfunc


GRB_CB_PRE_COLDEL = 1000
GRB_CB_PRE_ROWDEL = 1001
GRB_CB_PRE_SENCHG = 1002
GRB_CB_PRE_BNDCHG = 1003
GRB_CB_PRE_COECHG = 1004

GRB_CB_SPX_ITRCNT = 2000
GRB_CB_SPX_OBJVAL = 2001
GRB_CB_SPX_PRIMINF = 2002
GRB_CB_SPX_DUALINF = 2003
GRB_CB_SPX_ISPERT = 2004

GRB_CB_MIP_OBJBST = 3000
GRB_CB_MIP_OBJBND = 3001
GRB_CB_MIP_NODCNT = 3002
GRB_CB_MIP_SOLCNT = 3003
GRB_CB_MIP_CUTCNT = 3004
GRB_CB_MIP_NODLFT = 3005
GRB_CB_MIP_ITRCNT = 3006

GRB_CB_MIPSOL_SOL = 4001
GRB_CB_MIPSOL_OBJ = 4002
GRB_CB_MIPSOL_OBJBST = 4003
GRB_CB_MIPSOL_OBJBND = 4004
GRB_CB_MIPSOL_NODCNT = 4005
GRB_CB_MIPSOL_SOLCNT = 4006

GRB_CB_MIPNODE_STATUS = 5001
GRB_CB_MIPNODE_REL = 5002
GRB_CB_MIPNODE_OBJBST = 5003
GRB_CB_MIPNODE_OBJBND = 5004
GRB_CB_MIPNODE_NODCNT = 5005
GRB_CB_MIPNODE_SOLCNT = 5006

GRB_CB_MSG_STRING = 6001
GRB_CB_RUNTIME = 6002


class SolverGurobi(Solver):

    def __init__(self, model: Model, name: str, sense: str):
        super().__init__(model, name, sense)

        # setting class members to default values
        self._log = ""
        self._env = ffi.NULL
        self._model = ffi.NULL
        self._callback = None

        self._env = ffi.new("GRBenv **")

        # creating Gurobi environment
        st = GRBloadenv(self._env, ''.encode('utf-8'))
        if st != 0:
            raise Exception('Gurobi environment could not be loaded,\
check your license.')
        self._env = self._env[0]

        # creating Gurobi model
        self._model = ffi.new("GRBmodel **")
        st = GRBnewmodel(self._env, self._model,
                         name.encode('utf-8'), 0,
                         ffi.NULL, ffi.NULL, ffi.NULL, ffi.NULL, ffi.NULL)
        if st != 0:
            raise Exception('Could not create Gurobi model')
        self._model = self._model[0]

        # setting objective sense
        if sense == MAXIMIZE:
            self.set_int_attr("ModelSense", -1)
        else:
            self.set_int_attr("ModelSense", 1)

        # default number of threads
        self.__threads = 0

        # fine grained control of what is changed
        # for selective call on model.update
        self.__n_cols_buffer = 0
        self.__n_int_buffer = 0
        self.__n_rows_buffer = 0
        self.__n_modified_cols = 0
        self.__n_modified_rows = 0
        self.__updated = True
        self.__name_space = ffi.new('char[{}]'.format(MAX_NAME_SIZE))

    def __del__(self):
        # freeing Gurobi model and environment
        if self._model:
            GRBfreemodel(self._model)
        if self._env:
            GRBfreeenv(self._env)

    def add_var(self,
                obj: float = 0,
                lb: float = 0,
                ub: float = INF,
                var_type: str = CONTINUOUS,
                column: Column = None,
                name: str = ""):
        # collecting column data
        nz = 0 if column is None else len(column.constrs)
        if nz:
            self.flush_rows()
            vind = ffi.new("int[]", [c.idx for c in column.constrs])
            vval = ffi.new("double[]",
                           [column.coeffs[i] for i in range(nz)])
        else:
            vind = ffi.NULL
            vval = ffi.NULL

        if not name:
            name = 'x({})'.format(self.num_cols())

        # variable type
        vtype = var_type.encode('utf-8')

        st = GRBaddvar(self._model, nz,
                       vind, vval, obj, lb, ub,
                       vtype, name.encode('utf-8'))
        if st != 0:
            raise Exception('Error adding variable {} to model.'.format(name))

        self.__n_cols_buffer += 1
        if vtype == BINARY or vtype == INTEGER:
            self.__n_int_buffer += 1

    def add_constr(self, lin_expr: LinExpr, name: str = ""):
        self.flush_cols()

        # collecting linear expression data
        nz = len(lin_expr.expr)
        cind = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
        cval = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])

        # constraint sense and rhs
        sense = lin_expr.sense.encode('utf-8')
        rhs = -lin_expr.const

        if not name:
            name = 'r({})'.format(self.num_rows())

        st = GRBaddconstr(self._model, nz,
                          cind, cval,
                          sense, rhs, name.encode('utf-8'))
        if st != 0:
            raise Exception('Error adding constraint {} to the model'.format(
                name))
        self.__n_rows_buffer += 1

    def get_objective_bound(self) -> float:
        return self.get_dbl_attr("ObjBound")

    def get_objective(self) -> LinExpr:
        return self.get_dbl_attr("ObjVal")

    def get_objective_const(self) -> float:
        return self.get_dbl_attr("ObjCon")

    def relax(self):
        self.flush_cols()
        idxv = [var.idx for var in self.model.vars
                if var.var_type in [BINARY, INTEGER]]

        n = len(idxv)
        idxs = ffi.new("int[]", idxv)

        cont_char = CONTINUOUS.encode("utf-8")
        ccont = ffi.new("char[]", [cont_char for i in range(n)])

        attr = 'VType'.encode('utf-8')
        GRBsetcharattrlist(self._model, attr, n, idxs, ccont)
        self.__updated = False

    def get_max_seconds(self) -> float:
        return self.get_dbl_param("TimeLimit")

    def set_max_seconds(self, max_seconds: float):
        self.set_dbl_param("TimeLimit", max_seconds)

    def get_max_solutions(self) -> int:
        return self.get_int_param("SolutionLimit")

    def set_max_solutions(self, max_solutions: int):
        self.set_int_param("SolutionLimit", max_solutions)

    def get_max_nodes(self) -> int:
        rdbl = self.get_dbl_param("NodeLimit")
        rint = min(maxsize, int(rdbl))
        return rint

    def set_max_nodes(self, max_nodes: int):
        self.set_dbl_param("NodeLimit", float(max_nodes))

    def set_num_threads(self, threads: int):
        self.__threads = threads

    def optimize(self) -> OptimizationStatus:

        # todo add branch_selector and incumbent_updater callbacks
        @ffi.callback("""
           int (GRBmodel *, void *, int, void *)
        """)
        def callback(p_model: CData,
                     p_cbdata: CData,
                     where: int,
                     p_usrdata: CData) -> int:

            # adding cuts
            if self.model.cuts_generator and where == 5:  # MIPNODE == 5
                # obtaining relaxation solution and "translating" it
                cb_solution = ffi.new('double[{}]'.format(self.model.num_cols))
                GRBcbget(p_cbdata, where, GRB_CB_MIPNODE_REL, cb_solution)
                relax_solution = []
                for i in range(self.num_cols()):
                    if abs(cb_solution[i]) > 1e-8:
                        relax_solution.append((self.model.vars[i],
                                               cb_solution[i]))
                if len(relax_solution) == 0:
                    return 0

                # calling cuts generator
                cuts = self.model.cuts_generator.generate_cuts(relax_solution)
                # adding cuts
                for lin_expr in cuts:
                    # collecting linear expression data
                    nz = len(lin_expr.expr)
                    cind = ffi.new('int[]', [var.idx
                                             for var in lin_expr.expr.keys()])
                    cval = ffi.new('double[]',
                                   [coef for coef in lin_expr.expr.values()])

                    # constraint sense and rhs
                    sense = lin_expr.sense.encode('utf-8')
                    rhs = -lin_expr.const

                    GRBcbcut(p_cbdata, nz,
                             cind, cval, sense, rhs)

            # adding lazy constraints
            elif self.model.lazy_constrs_generator and where == 4:  # MIPSOL==4
                # obtaining relaxation solution and "translating" it
                cb_solution = ffi.new('double[{}]'.format(self.num_cols))
                GRBcbget(p_cbdata, where, GRB_CB_MIPSOL_SOL, cb_solution)
                solution = []
                for i in range(self.num_cols()):
                    if abs(cb_solution[i]) > 1e-8:
                        solution.append((self.model.vars[i], cb_solution[i]))

                # calling constraint generator
                lcg = self.model.lazy_constrs_generator
                constrs = lcg.generate_lazy_constrs(solution)
                # adding cuts
                for lin_expr in constrs:
                    # collecting linear expression data
                    nz = len(lin_expr.expr)
                    cind = ffi.new('int[]', [var.idx for
                                             var in lin_expr.expr.keys()])
                    cval = ffi.new('double[]', [c for c in
                                                lin_expr.expr.values()])
                    # constraint sense and rhs
                    sense = lin_expr.sense.encode('utf-8')
                    rhs = -lin_expr.const

                    GRBcblazy(p_cbdata, nz, cind, cval, sense, rhs)

            return 0

        self.update()
        if self.model.cuts_generator is not None:
            GRBsetcallbackfunc(self._model, callback, ffi.NULL)

        if self.__threads >= 1:
            self.set_int_param("Threads", self.__threads)

        self.set_int_param("Cuts", self.model.cuts)

        # executing Gurobi to solve the formulation
        status = GRBoptimize(self._model)
        if status == 10009:
            raise Exception('gurobi found but license not accepted,\
 please check it')

        status = self.get_int_attr("Status")
        # checking status for MIP optimization which
        # finished before the search to be
        # concluded (time, iteration limit...)
        if (self.num_int()):
            if status in [8, 9, 10, 11, 13]:
                nsols = self.get_int_attr("SolCount")
                if nsols >= 1:
                    return OptimizationStatus.FEASIBLE
                else:
                    return OptimizationStatus.NO_SOLUTION_FOUND

        # todo: read solution status (code below is incomplete)
        if status == 1:  # LOADED
            return OptimizationStatus.LOADED
        elif status == 2:  # OPTIMAL
            return OptimizationStatus.OPTIMAL
        elif status == 3:  # INFEASIBLE
            return OptimizationStatus.INFEASIBLE
        elif status == 4:  # INF_OR_UNBD
            return OptimizationStatus.UNBOUNDED
        elif status == 5:  # UNBOUNDED
            return OptimizationStatus.UNBOUNDED
        elif status == 6:  # CUTOFF
            return OptimizationStatus.CUTOFF
        elif status == 7:  # ITERATION_LIMIT
            return OptimizationStatus.OTHER
        elif status == 8:  # NODE_LIMIT
            return OptimizationStatus.OTHER
        elif status == 9:  # TIME_LIMIT
            return OptimizationStatus.OTHER
        elif status == 10:  # SOLUTION_LIMIT
            return OptimizationStatus.FEASIBLE
        elif status == 11:  # INTERRUPTED
            return OptimizationStatus.OTHER
        elif status == 12:  # NUMERIC
            return OptimizationStatus.OTHER
        elif status == 13:  # SUBOPTIMAL
            return OptimizationStatus.FEASIBLE
        elif status == 14:  # INPROGRESS
            return OptimizationStatus.OTHER
        elif status == 15:  # USER_OBJ_LIMIT
            return OptimizationStatus.FEASIBLE

        self._updated = True
        return status

    def get_objective_sense(self) -> str:
        isense = self.get_int_attr("ModelSense")
        if isense == 1:
            return MINIMIZE
        elif isense == -1:
            return MAXIMIZE
        else:
            raise Exception('Unknown sense')

    def set_objective_sense(self, sense: str):
        if sense.strip().upper() == MAXIMIZE.strip().upper():
            self.set_int_attr("ModelSense", -1)
        elif sense.strip().upper() == MINIMIZE.strip().upper():
            self.set_int_attr("ModelSense", 1)
        else:
            raise Exception("Unknown sense: {}, use {} or {}".format(sense,
                            MAXIMIZE, MINIMIZE))
        self.__updated = False

    def get_num_solutions(self) -> int:
        return self.get_int_attr("SolCount")

    def var_get_xi(self, var: Var, i: int) -> float:
        self.set_int_param("SolutionNumber", i)
        return self.get_dbl_attr_element("Xn", var.idx)

    def var_get_index(self, name: str) -> int:
        idx = ffi.new("int *")
        st = GRBgetvarbyname(self._model, name.encode('utf-8'), idx)
        if st:
            raise Exception("Error calling GRBgetvarbyname")
        return idx[0]

    def get_objective_value_i(self, i: int) -> float:
        self.set_int_param("SolutionNumber", i)
        return self.get_dbl_attr("PoolObjVal")

    def get_objective_value(self) -> float:
        return self.get_dbl_attr('ObjVal')

    def set_processing_limits(self,
                              max_time: float = INF,
                              max_nodes: float = INF,
                              max_sol: int = INF):
        # todo: Set limits even when they are 'inf'
        if max_time != INF:
            self.set_dbl_param("TimeLimit", max_time)
        if max_nodes != INF:
            self.set_dbl_param("NodeLimit", max_nodes)
        if max_sol != INF:
            self.set_int_param("SolutionLimit", max_sol)

    def set_objective(self, lin_expr: LinExpr, sense: str = "") -> None:
        # collecting linear expression data
        nz = len(lin_expr.expr)
        cind = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
        cval = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])

        # objective function constant
        const = lin_expr.const

        # resetting objective function
        num_vars = self.num_cols()
        zeros = ffi.new("double[]", [0.0 for i in range(num_vars)])

        attr = 'Obj'.encode('utf-8')
        st = GRBsetdblattrarray(self._model, attr,
                                0, num_vars, zeros)

        if st != 0:
            raise Exception('Could not set gurobi double attribute array Obj')

        # setting objective sense
        if sense == MAXIMIZE:
            self.set_int_attr("ModelSense", -1)
        elif sense == MINIMIZE:
            self.set_int_attr("ModelSense", 1)

        # setting objective function
        self.set_dbl_attr("ObjCon", const)
        error = GRBsetdblattrlist(self._model, attr, nz,
                                  cind, cval)
        if error != 0:
            raise Exception("Error modifying attribute Obj")
        self.__n_modified_cols += 1

    def set_objective_const(self, const: float) -> None:
        self.set_dbl_attr("ObjCon", const)
        self.__updated = False

    def set_start(self, start: List[Tuple[Var, float]]) -> None:
        # collecting data
        nz = len(start)
        cind = ffi.new("int[]", [el[0].idx for el in start])
        cval = ffi.new("double[]", [el[1] for el in start])

        st = GRBsetdblattrlist(self._model, 'Start'.encode('utf-8'), nz,
                               cind, cval)
        if st != 0:
            raise Exception("Error modifying attribute Start")
        self.__updated = False

    def flush_cols(self):
        """should be called in methods that require updated column
        information, e.g. when adding a new constraint"""
        if self.__n_cols_buffer or self.__n_modified_cols:
            self.update()

    def flush_rows(self):
        """should be called in methods that require updated row
        information, e.g. when adding a new column"""
        if self.__n_rows_buffer or self.__n_modified_rows:
            self.update()

    def write(self, file_path: str) -> None:
        # writing formulation to output file
        self.update()
        st = GRBwrite(self._model, file_path.encode('utf-8'))
        if st != 0:
            raise Exception('Could not write gurobi model.')

    def read(self, file_path: str) -> None:
        if not isfile(file_path):
            raise Exception('File {} does not exists'.format(file_path))
        GRBfreemodel(self._model)
        self._model = ffi.new('void **')
        st = GRBreadmodel(self._env, file_path.encode('utf-8'),
                          self._model)
        if st != 0:
            raise Exception('Could not read model {}, check contents'.format(
                            file_path))
        self._model = self._model[0]

    def num_cols(self) -> int:
        return self.get_int_attr("NumVars")+self.__n_cols_buffer

    def num_int(self) -> int:
        return self.get_int_attr("NumIntVars") + self.__n_int_buffer

    def num_rows(self) -> int:
        return self.get_int_attr("NumConstrs") + self.__n_rows_buffer

    def num_nz(self) -> int:
        self.flush_rows()
        return self.get_int_attr("NumNZs")

    def get_cutoff(self) -> float:
        return self.get_dbl_param("Cutoff")

    def set_cutoff(self, cutoff: float):
        self.set_dbl_param("Cutoff", cutoff)

    def get_mip_gap_abs(self) -> float:
        return self.get_dbl_param("MIPGapAbs")

    def set_mip_gap_abs(self, allowable_gap: float):
        self.set_dbl_param("MIPGapAbs", allowable_gap)

    def get_mip_gap(self) -> float:
        return self.get_dbl_param("MIPGap")

    def set_mip_gap(self, allowable_ratio_gap: float):
        self.set_dbl_param("MIPGap", allowable_ratio_gap)

    def get_verbose(self) -> int:
        return self.get_int_param("OutputFlag")

    def set_verbose(self, verbose: int):
        self.set_int_param("OutputFlag", verbose)

    def constr_get_expr(self, constr: Constr) -> LinExpr:
        self.flush_rows()

        nnz = ffi.new("int *")
        # obtaining number of non-zeros
        st = GRBgetconstrs(self._model, nnz, ffi.NULL, ffi.NULL, ffi.NULL,
                           constr.idx, 1)
        if st != 0:
            raise Exception('Could not get info for constraint {}'.format(
                            constr.idx))
        nz = nnz[0]

        # creating arrays to hold indices and coefficients
        cbeg = ffi.new("int[2]")
        cind = ffi.new("int[{}]".format(nz))
        cval = ffi.new("double[{}]".format(nz))

        # obtaining variables and coefficients
        st = GRBgetconstrs(self._model, nnz, cbeg, cind, cval,
                           constr.idx, 1)
        if st != 0:
            raise Exception.create('Could not query constraint contents')

        # obtaining sense and rhs
        c_sense = ffi.new('char *')
        rhs = ffi.new('double *')
        st = GRBgetcharattrelement(self._model, 'Sense'.encode('utf-8'),
                                   constr.idx, c_sense)
        if st != 0:
            raise Exception('Could not query sense for constraint {}'.format(
                            constr.idx))
        st = GRBgetdblattrelement(self._model, 'RHS'.encode('utf-8'),
                                  constr.idx, rhs)
        if st != 0:
            raise Exception('Could not query RHS for constraint {}'.format(
                            constr.idx))

        ssense = c_sense[0].decode('utf-8')
        # translating sense
        sense = ""
        if ssense == "<":
            sense = LESS_OR_EQUAL
        elif ssense == ">":
            sense = GREATER_OR_EQUAL
        elif ssense == "=":
            sense = EQUAL

        expr = LinExpr(const=-rhs[0], sense=sense)
        for i in range(nz):
            expr.add_var(self.model.vars[cind[i]], cval[i])

        return expr

    def constr_get_name(self, idx: int) -> str:
        self.flush_rows()
        return self.get_str_attr_element('ConstrName', idx)

    def constr_set_expr(self, constr: Constr, value: LinExpr) -> LinExpr:
        raise NotImplementedError("Gurobi functionality currently unavailable")

    def constr_get_pi(self, constr: "Constr") -> float:
        return self.get_dbl_attr("Pi", constr.idx)

    def constr_get_index(self, name: str) -> int:
        idx = ffi.new('int *')
        st = GRBgetconstrbyname(self._model, name.encode('utf-8'), idx)
        if st != 0:
            raise Exception("Error calling GRBgetconstrbyname")
        return idx[0]

    def var_get_lb(self, var: Var) -> float:
        self.flush_cols()
        return self.get_dbl_attr_element("LB", var.idx)

    def var_set_lb(self, var: Var, value: float) -> None:
        self.set_dbl_attr_element("LB", var.idx, value)
        self.__n_modified_cols += 1

    def var_get_ub(self, var: Var) -> float:
        self.flush_cols()
        return self.get_dbl_attr_element("UB", var.idx)

    def var_set_ub(self, var: Var, value: float) -> None:
        self.set_dbl_attr_element("UB", var.idx, value)
        self.__n_modified_cols += 1

    def var_get_obj(self, var: Var) -> float:
        self.flush_cols()
        return self.get_dbl_attr_element("Obj", var.idx)

    def var_set_obj(self, var: Var, value: float) -> None:
        self.set_dbl_attr_element("Obj", var.idx, value)
        self.__n_modified_cols += 1

    def var_get_var_type(self, var: Var) -> str:
        self.flush_cols()
        res = ffi.new('char *')
        st = GRBgetcharattrelement(self._model, 'VType'.encode('utf-8'),
                                   var.idx, res)
        if st != 0:
            raise Exception('Error querying variable type in gurobi')

        vt = res[0].decode('utf-8')

        if vt == 'B':
            return BINARY
        elif vt == "C":
            return CONTINUOUS
        elif vt == "I":
            return INTEGER

        raise ValueError("Gurobi: invalid variable type returned...")

    def var_set_var_type(self, var: Var, value: str) -> None:
        self.set_char_attr_element("VType", var.idx, value)
        self._updated = False

    def var_get_column(self, var: Var) -> Column:
        self.update()

        nnz = ffi.new('int*')

        # obtaining number of non-zeros
        error = GRBgetvars(self._model, nnz, ffi.NULL, ffi.NULL, ffi.NULL,
                           var.idx, 1)
        if error != 0:
            raise Exception('Error querying gurobi model information')

        nz = nnz[0]

        # creating arrays to hold indices and coefficients
        cbeg = ffi.new('int[2]')
        cind = ffi.new('int[{}]'.format(nz))
        cval = ffi.new('double[{}]'.format(nz))

        # obtaining variables and coefficients
        error = GRBgetvars(self._model, nnz, cbeg, cind, cval,
                           var.idx, 1)
        if error != 0:
            raise Exception('Error querying gurobi model information')

        constr = [self.model.constrs[cind[i]] for i in range(nz)]
        coefs = [float(cval[i]) for i in range(nz)]

        return Column(constr, coefs)

    def var_set_column(self, var: Var, value: Column):
        raise NotImplementedError("Gurobi functionality currently unavailable")

    def var_get_rc(self, var: Var) -> float:
        return self.get_dbl_attr_element("RC", var.idx)

    def var_get_x(self, var: Var) -> float:
        return self.get_dbl_attr_element("X", var.idx)

    def var_get_name(self, idx: int) -> str:
        self.flush_cols()
        return self.get_str_attr_element('VarName', idx)

    def get_emphasis(self) -> SearchEmphasis:
        fc = self.get_int_param("MIPFocus")
        if fc == 1:
            return SearchEmphasis.FEASIBILITY
        elif fc == 3 or fc == 2:
            return SearchEmphasis.OPTIMALITY

        return 0

    def set_emphasis(self, emph: SearchEmphasis):
        if emph == SearchEmphasis.FEASIBILITY:
            self.set_int_param("MIPFocus", 1)
        elif emph == SearchEmphasis.OPTIMALITY:
            self.set_int_param("MIPFocus", 2)
        else:
            self.set_int_param("MIPFocus", 0)

    def update(self):
        if ((self.__n_cols_buffer + self.__n_int_buffer +
             self.__n_rows_buffer +
             self.__n_modified_cols + self.__n_modified_rows) == 0
                and self.__updated):
            return
        GRBupdatemodel(self._model)
        self.__n_cols_buffer = 0
        self.__n_int_buffer = 0
        self.__n_rows_buffer = 0
        self.__n_modified_cols = 0
        self.__n_modified_rows = 0
        self.__updated = True

    def set_char_attr_element(self, name: str, index: int, value: str):
        assert len(value) == 1
        error = GRBsetcharattrelement(self._model, name.encode('utf-8'),
                                      index, value.encode('utf-8'))
        if error != 0:
            raise Exception(
                'Error setting gurobi char attr element {} index {} to value'.
                format(name, index, value))

    def get_dbl_attr_element(self, name: str, index: int) -> float:
        res = ffi.new('double *')
        error = GRBgetdblattrelement(self._model, name.encode('utf-8'),
                                     index, res)
        if error != 0:
            raise Exception('Error get grb double attr element {} index {}'.
                            format(name, index))
        return res[0]

    def set_dbl_attr_element(self, name: str, index: int, value: float):
        error = GRBsetdblattrelement(self._model, name.encode('utf-8'),
                                     index, value)
        if error != 0:
            raise Exception(
                "Error modifying dbl attribute {} for element {} to value {}".
                format(name, index, value))

    def set_int_attr(self, name: str, value: int):
        error = GRBsetintattr(self._model, name.encode('utf-8'), value)
        if error != 0:
            raise Exception("Error modifying int attribute {} to {}".
                            format(name, value))

    def set_dbl_attr(self, name: str, value: float):
        error = GRBsetdblattr(self._model, name.encode('utf-8'), value)
        if error != 0:
            raise Exception("Error modifying double attribute {} to {}".
                            format(name, value))

    def get_int_attr(self, name: str) -> int:
        res = ffi.new('int *')
        error = GRBgetintattr(self._model, name.encode('utf-8'), res)
        if error != 0:
            raise Exception('Error getting int attribute {}'.format(name))
        return res[0]

    def get_int_param(self, name: str) -> int:
        res = ffi.new('int *')
        error = GRBgetintparam(self._model, name.encode('utf-8'), res)
        if error != 0:
            raise Exception("Error getting gurobi integer parameter {}".
                            format(name))
        return res.value

    def set_int_param(self, name: str, value: int):
        error = GRBsetintparam(self._env,
                               name.encode('utf-8'), value)
        if error != 0:
            raise Exception("Error mofifying int parameter {} to value {}".
                            format(name, value))

    def get_dbl_attr(self, attr: str) -> float:
        res = ffi.new('double *')
        error = GRBgetdblattr(self._model, attr.encode('utf-8'), res)
        if error != 0:
            raise Exception('Error getting gurobi double attribute {}'.
                            format(attr))
        return res[0]

    def set_dbl_param(self, param: str, value: float):
        error = GRBsetdblparam(self._env, param.encode('utf-8'),
                               value)
        if error != 0:
            raise Exception("Error setting gurobi double param " +
                            param + " to {}".format(value))

    def get_dbl_param(self, param: str) -> float:
        res = ffi.new('double *')
        error = GRBgetdblparam(self._env, param.encode('utf-8'),
                               res)
        if error != 0:
            raise Exception("Error getting gurobi double parameter {}".
                            format(param))
        return res[0]

    def get_str_attr_element(self, attr: str, index: int) -> str:
        vName = ffi.new('char **')
        error = GRBgetstrattrelement(self._model, attr.encode('utf-8'), index,
                                     vName)
        if error != 0:
            raise Exception('Error getting str attribute {} index {}'.
                            format(attr, index))
        return ffi.string(vName[0]).decode('utf-8')
