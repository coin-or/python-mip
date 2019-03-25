from mip.model import *
from ctypes import *
from ctypes.util import *
from math import inf


class SolverGurobi(Solver):

    def __init__(self, model: Model, name: str, sense: str):
        super().__init__(model, name, sense)

        # setting class members to default values
        self._updated = False
        self._num_vars = 0
        self._num_constrs = 0
        self._log = ""
        self._env = c_void_p(0)
        self._model = c_void_p(0)
        self._callback = None

        # setting variables for an empty model
        numvars = c_int(0)
        obj = c_double(0)
        lb = c_double(0)
        ub = c_double(0)
        vtype = c_char_p()
        varnames = c_void_p(0)

        # creating Gurobi environment
        if GRBloadenv(byref(self._env), c_str(self._log)) != 0:
            # todo: raise exception when environment can't be loaded
            pass

        # creating Gurobi model
        if GRBnewmodel(self._env, byref(self._model), c_str(name), numvars,
                       byref(obj), byref(lb), byref(ub), vtype, varnames) != 0:
            # todo: raise exception when environment can't be generated
            pass

        # setting objective sense
        if sense == MAXIMIZE:
            GRBsetintattr(self._model, c_str("ModelSense"), -1)
        elif sense == MINIMIZE:
            GRBsetintattr(self._model, c_str("ModelSense"), 1)

    def __del__(self):
        # freeing Gurobi model and environment
        if self._model:
            GRBfreemodel(self._model)
        if self._env:
            GRBfreeenv(self._env)

    def add_var(self,
                obj: float = 0,
                lb: float = 0,
                ub: float = float("inf"),
                type: str = "C",
                column: "Column" = None,
                name: str = "") -> int:
        # collecting column data
        numnz = 0 if column is None else len(column.constrs)
        vind = (c_int * numnz)()
        vval = (c_double * numnz)()

        # collecting column coefficients
        for i in range(numnz):
            vind[i] = column.constrs[i].idx
            vval[i] = column.coeffs[i]

        # variable type
        vtype = c_char(ord(type))

        # variable index
        idx = self._num_vars
        self._num_vars += 1

        GRBaddvar(self._model, c_int(numnz), vind, vval, c_double(obj), c_double(lb), c_double(ub),
                  vtype, c_str(name))
        self._updated = False

        return idx

    def add_constr(self, lin_expr: LinExpr, name: str = "") -> int:
        # collecting linear expression data
        numnz = len(lin_expr.expr)
        cind = (c_int * numnz)()
        cval = (c_double * numnz)()

        # collecting variable coefficients
        for i, (var, coeff) in enumerate(lin_expr.expr.items()):
            cind[i] = var.idx
            cval[i] = coeff

        # constraint sense and rhs
        sense = c_char(ord(lin_expr.sense))
        rhs = c_double(-lin_expr.const)

        # constraint index
        idx = self._num_constrs
        self._num_constrs += 1

        GRBaddconstr(self._model, numnz, cind, cval, sense, rhs, c_str(name))
        self._updated = False

        return idx

    def get_objective_bound(self) -> float:
        res = c_double()
        GRBgetdblattr(self._model, c_str("ObjBound"), byref(res))
        return res.value

    def get_objective(self) -> LinExpr:
        res = c_double()
        GRBgetdblattr(self._model, c_str("ObjVal"), byref(res))
        return res.value

    def get_objective_const(self) -> float:
        res = c_double()
        GRBgetdblattr(self._model, c_str("ObjCon"), byref(res))
        return res.value

    def relax(self):
        idxs = list()
        for var in self.model.vars:
            vtype = self.var_get_type(var)
            if vtype == BINARY or vtype == INTEGER:
                idxs.append(var.idx)

        ccont = (c_char * len(idxs))()
        for i in range(len(idxs)):
            ccont[i] = CONTINUOUS.encode("utf-8")

        GRBsetcharattrarray(self._model, c_str("VType"), 0, len(idxs), ccont)
        GRBupdatemodel(self._model)

    def get_max_seconds(self) -> float:
        res = c_double()
        GRBgetdblparam(GRBgetenv(self._model), c_str("TimeLimit"), byref(res))
        return res.value

    def set_max_seconds(self, max_seconds: float):
        st = GRBsetdblparam(GRBgetenv(self._model), c_str("TimeLimit"), c_double(max_seconds))
        assert st == 0

    def get_max_solutions(self) -> int:
        res = c_double()
        st = GRBgetintparam(GRBgetenv(self._model), c_str("SolutionLimit"), byref(res))
        assert st == 0
        return res.value

    def set_max_solutions(self, max_solutions: int):
        st = GRBsetintparam(GRBgetenv(self._model), c_str("SolutionLimit"), c_int(max_solutions))
        assert st == 0

    def get_max_nodes(self) -> int:
        res = c_double()
        st = GRBgetdblparam(GRBgetenv(self._model), c_str("NodeLimit"), byref(res))
        assert st == 0
        rdbl = res.value
        rint = min(sys.maxsize, int(rdbl))
        return rint

    def set_max_nodes(self, max_nodes: int):
        st = GRBsetdblparam(GRBgetenv(self._model), c_str("NodeLimit"), c_double(max_nodes))
        assert st == 0

    def optimize(self) -> int:
        # todo add branch_selector and incumbent_updater callbacks
        def callback(p_model: c_void_p,
                     p_cbdata: c_void_p,
                     where: int,
                     p_usrdata: c_void_p) -> int:
            if self.model.cut_generators and where == 5:  # MIPNODE == 5
                # obtaining relaxation solution and "translating" it
                solution = (c_double * self._num_vars)()
                GRBcbget(p_cbdata, where, GRB_CB_MIPNODE_REL, solution)
                relax_solution = []
                for i in range(self._num_vars):
                    if solution[i] <= -EPS or solution[i] >= EPS:
                        relax_solution.append((self.model.vars[i], solution[i]))

                # calling cut generators
                for cg in self.model.cut_generators:
                    cuts = cg.generate_cuts(relax_solution)
                    # adding cuts
                    for lin_expr in cuts:
                        # collecting linear expression data
                        numnz = len(lin_expr.expr)
                        cind = (c_int * numnz)()
                        cval = (c_double * numnz)()

                        # collecting variable coefficients
                        for i, (var, coeff) in enumerate(lin_expr.expr.items()):
                            cind[i] = var.idx
                            cval[i] = coeff

                        # constraint sense and rhs
                        sense = c_char(ord(lin_expr.sense))
                        rhs = c_double(-lin_expr.const)

                        GRBcbcut(p_cbdata, numnz, cind, cval, sense, rhs)

            return 0

        if self.model.cut_generators:
            self._callback = GRBcallbacktype(callback)
            GRBsetcallbackfunc(self._model, self._callback, c_void_p(0))

        # executing Gurobi to solve the formulation
        status = int(GRBoptimize(self._model))
        if status == 10009:
            raise Exception('gurobi found but license not accepted, please check it')
        
        status = c_int(0)
        st = GRBgetintattr(self._model, c_str("Status"), byref(status))
        if (st):
            raise Exception('could not check optimization status')
        
        status = status.value
        
        # todo: read solution status (code below is incomplete)
        if status == 1:  # LOADED
            return LOADED
        elif status == 2:  # OPTIMAL
            return OPTIMAL
        elif status == 3:  # INFEASIBLE
            return INFEASIBLE
        elif status == 4:  # INF_OR_UNBD
            return UNBOUNDED
        elif status == 5:  # UNBOUNDED
            return UNBOUNDED
        elif status == 6:  # CUTOFF
            return CUTOFF
        elif status == 7:  # ITERATION_LIMIT
            return -10000
        elif status == 8:  # NODE_LIMIT
            return -10000
        elif status == 9:  # TIME_LIMIT
            return -10000
        elif status == 10:  # SOLUTION_LIMIT
            return FEASIBLE
        elif status == 11:  # INTERRUPTED
            return -10000
        elif status == 12:  # NUMERIC
            return -10000
        elif status == 13:  # SUBOPTIMAL
            return FEASIBLE
        elif status == 14:  # INPROGRESS
            return -10000
        elif status == 15:  # USER_OBJ_LIMIT
            return FEASIBLE

        self._updated = True
        return status

    def get_objective_sense(self) -> str:
        isense = c_int(0)
        st = GRBgetintattr(self._model, c_str("ModelSense"), byref(isense))
        assert st == 0
        if isense.value == 1:
            return MINIMIZE
        elif isense.value == -1:
            return MAXIMIZE
        else:
            raise Exception('Unknown sense')

    def set_objective_sense(self, sense: str):
        if sense.strip().upper() == MAXIMIZE.strip().upper():
            st = GRBsetintattr(self._model, c_str("ModelSense"), c_int(-1))
            assert st == 0
        elif sense.strip().upper() == MINIMIZE.strip().upper():
            st = GRBsetintattr(self._model, c_str("ModelSense"), c_int(1))
            assert st == 0
        else:
            raise Exception("Unknown sense: {}, use {} or {}".format(sense,
                                                                     MAXIMIZE,
                                                                     MINIMIZE))

    def get_num_solutions(self) -> int:
        res = c_int(0)
        st = GRBgetintattr(self._model, c_str("SolCount"), byref(res))
        assert st == 0
        return res.value

    def var_get_xi(self, var: "Var", i: int) -> float:
        res = c_double()
        st = GRBsetintparam(GRBgetenv(self._model), c_str("SolutionNumber"),
                            c_int(i))
        assert st == 0
        st = GRBgetdblattrelement(self._model, c_str("Xn"),
                                  c_int(var.idx), byref(res))
        assert st == 0
        return res.value

    def get_objective_value_i(self, i: int) -> float:
        res = c_double(0)
        st = GRBgetdblattr(self._model, c_str("PoolObjVal"), byref(res))
        assert st == 0
        return res.value

    def get_objective_value(self) -> float:
        res = c_double(float('inf'))
        st = GRBgetdblattr(self._model, c_str('ObjVal'), byref(res))
        assert st == 0
        return res.value

    def set_processing_limits(self,
                              max_time: float = inf,
                              max_nodes: float = inf,
                              max_sol: int = inf):
        # todo: Set limits even when they are 'inf'
        if max_time != inf:
            res = GRBsetdblparam(GRBgetenv(self._model), c_str("TimeLimit"), c_double(max_time))
            assert res == 0
        if max_nodes != inf:
            res = GRBsetdblparam(GRBgetenv(self._model), c_str("NodeLimit"), c_double(max_nodes))
            assert res == 0
        if max_sol != inf:
            res = GRBsetintparam(GRBgetenv(self._model), c_str("SolutionLimit"), c_int(max_sol))
            assert res == 0

    def set_objective(self, lin_expr: "LinExpr", sense: str = "") -> None:
        # collecting linear expression data
        numnz = len(lin_expr.expr)
        cind = (c_int * numnz)()
        cval = (c_double * numnz)()

        # collecting variable coefficients
        for i, (var, coeff) in enumerate(lin_expr.expr.items()):
            cind[i] = var.idx
            cval[i] = coeff

        # objective function constant
        const = c_double(lin_expr.const)

        # resetting objective function
        num_vars = c_int(self._num_vars)
        zeros = (c_double * self._num_vars)()
        for i in range(self._num_vars):
            zeros[i] = 0.0
        GRBsetdblattrarray(self._model, c_str("Obj"), c_int(0), num_vars, zeros)

        # setting objective sense
        if sense == MAXIMIZE:
            GRBsetintattr(self._model, c_str("ModelSense"), -1)
        elif sense == MINIMIZE:
            GRBsetintattr(self._model, c_str("ModelSense"), 1)

        # setting objective function
        GRBsetdblattr(self._model, c_str("ObjCon"), const)
        GRBsetdblattrlist(self._model, c_str("Obj"), c_int(numnz), cind, cval)
        self._updated = False

    def set_objective_const(self, const: float) -> None:
        GRBsetdblattr(self._model, c_str("ObjCon"), c_double(const))

    def set_start(self, start: List[Tuple["Var", float]]) -> None:
        # collecting data
        numnz = len(start)
        cind = (c_int * numnz)()
        cval = (c_double * numnz)()

        # collecting variable coefficients
        for i in range(len(start)):
            cind[i] = start[i][0].idx
            cval[i] = start[i][1]

        GRBsetdblattrlist(self._model, c_str("Start"), numnz, cind, cval)
        self._updated = False

    def update(self) -> None:
        GRBupdatemodel(self._model)
        self._updated = True

    def write(self, file_path: str) -> None:
        # writing formulation to output file
        if not self._updated:
            self.update()
        GRBwrite(self._model, c_str(file_path))

    def read(self, file_path: str) -> None:
        GRBfreemodel(self._model)
        self._model = c_void_p(0)
        GRBreadModel(self._env, c_str(file_path), byref(self._model))

    def num_cols(self) -> int:
        res = c_int(0)
        st = GRBgetintattr(self._model, c_str('NumVars'), byref(res))
        assert st == 0
        return res.value

    def num_rows(self) -> int:
        res = c_int(0)
        st = GRBgetintattr(self._model, c_str('NumConstrs'), byref(res))
        assert st == 0
        return res.value

    def num_nz(self) -> int:
        res = c_int(0)
        st = GRBgetintattr(self._model, c_str('NumNZs'), byref(res))
        assert st == 0
        return res.value

    def get_cutoff(self) -> float:
        res = c_double(0.0)
        st = GRBgetdblparam(GRBgetenv(self._model), c_str("Cutoff"), byref(res))
        assert st == 0
        return res.value

    def set_cutoff(self, cutoff: float):
        st = GRBsetdblparam(GRBgetenv(self._model), c_str("Cutoff"), c_double(cutoff))
        assert st == 0

    def get_mip_gap_abs(self) -> float:
        res = c_double(0.0)
        st = GRBgetdblparam(GRBgetenv(self._model), c_str("MIPGapAbs"), byref(res))
        assert st == 0
        return res.value

    def set_mip_gap_abs(self, allowable_gap: float):
        st = GRBsetdblparam(GRBgetenv(self._model), c_str("MIPGapAbs"),
                            c_double(allowable_gap))
        assert st == 0

    def get_mip_gap(self) -> float:
        res = c_double(0.0)
        st = GRBgetdblparam(GRBgetenv(self._model), c_str("MIPGap"), byref(res))
        assert st == 0
        return res.value

    def set_mip_gap(self, allowable_ratio_gap: float):
        st = GRBsetdblparam(GRBgetenv(self._model), c_str("MIPGap"),
                            c_double(allowable_ratio_gap))
        assert st == 0

    def constr_get_expr(self, constr: Constr) -> LinExpr:
        if not self._updated:
            self.update()

        numnz = c_int()
        cbeg = POINTER(c_int)()
        cind = POINTER(c_int)()
        cval = POINTER(c_double)()

        # obtaining number of non-zeros
        GRBgetconstrs(self._model, byref(numnz), cbeg, cind, cval, c_int(constr.idx), c_int(1))

        # creating arrays to hold indices and coefficients
        cbeg = (c_int * 2)()  # beginning and ending
        cind = (c_int * numnz.value)()
        cval = (c_double * numnz.value)()

        # obtaining variables and coefficients
        GRBgetconstrs(self._model, byref(numnz), cbeg, cind, cval, c_int(constr.idx), c_int(1))

        # obtaining sense and rhs
        c_sense = c_char()
        rhs = c_double()
        GRBgetcharattrelement(self._model, c_str("Sense"), c_int(constr.idx), byref(c_sense))
        GRBgetdblattrelement(self._model, c_str("RHS"), c_int(constr.idx), byref(rhs))

        # translating sense
        sense = ""
        if c_sense.value == b"<":
            sense = LESS_OR_EQUAL
        elif c_sense.value == b">":
            sense = GREATER_OR_EQUAL
        elif c_sense.value == b"=":
            sense = EQUAL

        expr = LinExpr(const=-rhs.value, sense=sense)
        for i in range(numnz.value):
            expr.add_var(self.model.vars[cind[i]], cval[i])

        return expr

    def constr_get_name(self, idx: int) -> str:
        vName = c_char_p(0)
        st = GRBgetstrattrelement(self._model, c_str('ConstrName'), c_int(idx), byref(vName))
        assert st == 0
        return vName.value.decode('utf-8')

    def constr_set_expr(self, constr: Constr, value: LinExpr) -> LinExpr:
        raise NotImplementedError("Gurobi: functionality currently unavailable via PyMIP...")

    def constr_get_pi(self, constr: "Constr") -> float:
        res = c_double()
        st = GRBgetdblattrelement(self._model, c_str("Pi"), c_int(constr.idx), byref(res))
        assert st == 0
        return res.value

    def var_get_lb(self, var: "Var") -> float:
        if not self._updated:
            self.update()

        res = c_double()
        st = GRBgetdblattrelement(self._model, c_str("LB"), c_int(var.idx), byref(res))
        assert st == 0
        return res.value

    def var_set_lb(self, var: "Var", value: float) -> None:
        GRBsetdblattrelement(self._model, c_str("LB"), c_int(var.idx), c_double(value))
        self._updated = False

    def var_get_ub(self, var: "Var") -> float:
        if not self._updated:
            self.update()

        res = c_double()
        GRBgetdblattrelement(self._model, c_str("UB"), c_int(var.idx), byref(res))
        return res.value

    def var_set_ub(self, var: "Var", value: float) -> None:
        GRBsetdblattrelement(self._model, c_str("UB"), c_int(var.idx), c_double(value))
        self._updated = False

    def var_get_obj(self, var: "Var") -> float:
        if not self._updated:
            self.update()

        res = c_double()
        GRBgetdblattrelement(self._model, c_str("Obj"), c_int(var.idx), byref(res))
        return res.value

    def var_set_obj(self, var: "Var", value: float) -> None:
        GRBsetdblattrelement(self._model, c_str("Obj"), c_int(var.idx), c_double(value))
        self._updated = False

    def var_get_type(self, var: "Var") -> str:
        if not self._updated:
            self.update()

        res = c_char(0)
        GRBgetcharattrelement(self._model, c_str("VType"), c_int(var.idx), byref(res))

        if res.value == b"B":
            return BINARY
        elif res.value == b"C":
            return CONTINUOUS
        elif res.value == b"I":
            return INTEGER

        raise ValueError("Gurobi: invalid variable type returned...")

    def var_set_type(self, var: "Var", value: str) -> None:
        if value == BINARY:
            vtype = c_char(ord("B"))
        elif value == CONTINUOUS:
            vtype = c_char(ord("C"))
        elif value == INTEGER:
            vtype = c_char(ord("I"))
        else:
            raise ValueError("Gurobi: invalid variable type...")

        GRBsetcharattrelement(self._model, c_str("VType"), c_int(var.idx), vtype)
        self._updated = False

    def var_get_column(self, var: "Var"):
        if not self._updated:
            self.update()

        numnz = c_int()
        cbeg = POINTER(c_int)()
        cind = POINTER(c_int)()
        cval = POINTER(c_double)()

        # obtaining number of non-zeros
        GRBgetvars(self._model, byref(numnz), cbeg, cind, cval, c_int(var.idx), c_int(1))

        # creating arrays to hold indices and coefficients
        cbeg = (c_int * 2)()  # beginning and ending
        cind = (c_int * numnz.value)()
        cval = (c_double * numnz.value)()

        # obtaining variables and coefficients
        GRBgetvars(self._model, byref(numnz), cbeg, cind, cval, c_int(var.idx), c_int(1))

        constr = [self.model.constrs[cind[i]] for i in range(numnz.value)]
        coefs = [float(cval[i]) for i in range(numnz.value)]

        col = Column(constr, coefs)

        return col

    def var_set_column(self, var: "Var", value: Column):
        raise NotImplementedError("Gurobi: functionality currently unavailable via PyMIP...")

    def var_get_rc(self, var: "Var") -> float:
        res = c_double()
        GRBgetdblattrelement(self._model, c_str("RC"), c_int(var.idx), byref(res))
        return res.value

    def var_get_x(self, var: Var) -> float:
        res = c_double()
        GRBgetdblattrelement(self._model, c_str("X"), c_int(var.idx), byref(res))
        return res.value

    def var_get_name(self, idx: int) -> str:
        vName = c_char_p(0)
        st = GRBgetstrattrelement(self._model, c_str('VarName'), c_int(idx), byref(vName))
        assert st == 0
        return vName.value.decode('utf-8')

    def get_emphasis(self) -> int:
        fc = c_int(0)
        st = GRBgetintparam(GRBgetenv(self._model), c_str("MIPFocus"),
                            byref(fc))
        assert st == 0
        if fc == 1:
            return FEASIBILITY
        elif fc == 3 or fc == 2:
            return OPTIMALITY

        return 0

    def set_emphasis(self, emph: int):
        if emph == FEASIBILITY:
            st = GRBsetintparam(GRBgetenv(self._model), c_str("MIPFocus"),
                                c_int(1))
            assert st == 0
        elif emph == OPTIMALITY:
            st = GRBsetintparam(GRBgetenv(self._model), c_str("MIPFocus"),
                                c_int(2))
            assert st == 0
        else:
            st = GRBsetintparam(GRBgetenv(self._model), c_str("MIPFocus"),
                                c_int(0))
            assert st == 0


# auxiliary functions
def c_str(value) -> c_char_p:
    """
    This function converts a python string into a C compatible char[]
    :param value: input string
    :return: string converted to C"s format
    """
    return create_string_buffer(value.encode("utf-8"))


has_gurobi = False

try:
    found = False
    libPath = None

    for majorVersion in reversed(range(2, 12)):
        for minorVersion in reversed(range(0, 11)):
            try:
                libPath = find_library('gurobi{}{}'.format(majorVersion, minorVersion))
                if libPath != None:
                    break
            except:
                continue
        if libPath != None:
            break

    if libPath == None:
        raise Exception()
    grblib = CDLL(libPath)
    print('gurobi version {}.{} found'.format(majorVersion, minorVersion))
    has_gurobi = True
except:
    has_gurobi = False
# create/release environment and model

if has_gurobi:
    GRBloadenv = grblib.GRBloadenv
    GRBloadenv.restype = c_int
    GRBloadenv.argtypes = [c_void_p, c_char_p]

    GRBnewmodel = grblib.GRBnewmodel
    GRBnewmodel.restype = c_int
    GRBnewmodel.argtypes = [c_void_p, c_void_p, c_char_p, c_int, POINTER(c_double), POINTER(c_double),
                            POINTER(c_double), c_char_p, c_void_p]

    GRBfreeenv = grblib.GRBfreeenv
    GRBfreeenv.restype = c_int
    GRBfreeenv.argtypes = [c_void_p]

    GRBfreemodel = grblib.GRBfreemodel
    GRBfreemodel.argtypes = [c_void_p]

    # manipulate attributes

    GRBgetintattr = grblib.GRBgetintattr
    GRBgetintattr.restype = c_int
    GRBgetintattr.argtypes = [c_void_p, c_char_p, POINTER(c_int)]

    GRBsetintattr = grblib.GRBsetintattr
    GRBsetintattr.restype = c_int
    GRBsetintattr.argtypes = [c_void_p, c_char_p, c_int]

    GRBgetintattrelement = grblib.GRBgetintattrelement
    GRBgetintattrelement.restype = c_int
    GRBgetintattrelement.argtypes = [c_void_p, c_char_p, c_int, POINTER(c_int)]

    GRBsetintattrelement = grblib.GRBsetintattrelement
    GRBsetintattrelement.restype = c_int
    GRBsetintattrelement.argtypes = [c_void_p, c_char_p, c_int, c_int]

    GRBgetdblattr = grblib.GRBgetdblattr
    GRBgetdblattr.restype = c_int
    GRBgetdblattr.argtypes = [c_void_p, c_char_p, POINTER(c_double)]

    GRBsetdblattr = grblib.GRBsetdblattr
    GRBsetdblattr.restype = c_int
    GRBsetdblattr.argtypes = [c_void_p, c_char_p, c_double]

    GRBgetdblattrarray = grblib.GRBgetdblattrarray
    GRBgetdblattrarray.restype = c_int
    GRBgetdblattrarray.argtypes = [c_void_p, c_char_p, c_int, c_int, POINTER(c_double)]

    GRBsetdblattrarray = grblib.GRBsetdblattrarray
    GRBsetdblattrarray.restype = c_int
    GRBsetdblattrarray.argtypes = [c_void_p, c_char_p, c_int, c_int, POINTER(c_double)]

    GRBsetdblattrlist = grblib.GRBsetdblattrlist
    GRBsetdblattrlist.restype = c_int
    GRBsetdblattrlist.argtypes = [c_void_p, c_char_p, c_int, POINTER(c_int), POINTER(c_double)]

    GRBgetdblattrelement = grblib.GRBgetdblattrelement
    GRBgetdblattrelement.restype = c_int
    GRBgetdblattrelement.argtypes = [c_void_p, c_char_p, c_int, POINTER(c_double)]

    GRBsetdblattrelement = grblib.GRBsetdblattrelement
    GRBsetdblattrelement.restype = c_int
    GRBsetdblattrelement.argtypes = [c_void_p, c_char_p, c_int, c_double]

    GRBsetcharattrarray = grblib.GRBsetcharattrarray
    GRBsetcharattrarray.restype = c_int
    GRBsetcharattrarray.argtypes = [c_void_p, c_char_p, c_int, c_int, c_char_p]

    GRBgetcharattrelement = grblib.GRBgetcharattrelement
    GRBgetcharattrelement.restype = c_int
    GRBgetcharattrelement.argtypes = [c_void_p, c_char_p, c_int, POINTER(c_char)]

    GRBsetcharattrelement = grblib.GRBsetcharattrelement
    GRBsetcharattrelement.restype = c_int
    GRBsetcharattrelement.argtypes = [c_void_p, c_char_p, c_int, c_char]

    GRBgetstrattrelement = grblib.GRBgetstrattrelement
    GRBgetstrattrelement.argtypes = [c_void_p, c_char_p, c_int, POINTER(c_char_p)]
    GRBgetstrattrelement.restype = c_int

    # manipulate parameter(s)

    GRBgetintparam = grblib.GRBgetintparam
    GRBgetintparam.argtypes = [c_void_p, c_char_p, POINTER(c_int)]
    GRBgetintparam.restype = c_int

    GRBsetintparam = grblib.GRBsetintparam
    GRBsetintparam.argtypes = [c_void_p, c_char_p, c_int]
    GRBsetintparam.restype = c_int

    GRBgetdblparam = grblib.GRBgetdblparam
    GRBgetdblparam.argtypes = [c_void_p, c_char_p, POINTER(c_double)]
    GRBgetdblparam.restype = c_int

    GRBsetdblparam = grblib.GRBsetdblparam
    GRBsetdblparam.argtypes = [c_void_p, c_char_p, c_double]
    GRBsetdblparam.restype = c_int

    # manipulate objective function(s)

    GRBsetobjectiven = grblib.GRBsetobjectiven
    GRBsetobjectiven.restype = c_int
    GRBsetobjectiven.argtypes = [c_void_p, c_int, c_int, c_double, c_double, c_double, c_char_p,
                                 c_double, c_int, POINTER(c_int), POINTER(c_double)]

    # add variables and constraints

    GRBaddvar = grblib.GRBaddvar
    GRBaddvar.restype = c_int
    GRBaddvar.argtypes = [c_void_p, c_int, POINTER(c_int), POINTER(c_double), c_double, c_double,
                          c_double, c_char, c_char_p]

    GRBaddconstr = grblib.GRBaddconstr
    GRBaddconstr.restype = c_int
    GRBaddconstr.argtypes = [c_void_p, c_int, POINTER(c_int), POINTER(c_double), c_char, c_double,
                             c_char_p]

    # get constraints

    GRBgetconstrs = grblib.GRBgetconstrs
    GRBgetconstrs.restype = c_int
    GRBgetconstrs.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int), POINTER(c_int),
                              POINTER(c_double), c_int, c_int]

    # get variables
    GRBgetvars = grblib.GRBgetvars
    GRBgetvars.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int),
                           POINTER(c_int), POINTER(c_double), c_int, c_int]
    GRBgetvars.restype = c_int

    # callback functions and constants

    GRBcallbacktype = CFUNCTYPE(c_int, c_void_p, c_void_p, c_int, c_void_p)

    GRBsetcallbackfunc = grblib.GRBsetcallbackfunc
    GRBsetcallbackfunc.restype = c_int
    GRBsetcallbackfunc.argtypes = [c_void_p, GRBcallbacktype, c_void_p]

    GRBcbcut = grblib.GRBcbcut
    GRBcbcut.restype = c_int
    GRBcbcut.argtypes = [c_void_p, c_int, POINTER(c_int), POINTER(c_double), c_char, c_double]

    GRBcbget = grblib.GRBcbget
    GRBcbget.restype = c_int
    GRBcbget.argtypes = [c_void_p, c_int, c_int, c_void_p]

    GRBcbsolution = grblib.GRBcbsolution
    GRBcbsolution.restype = c_int
    GRBcbsolution.argtypes = [c_void_p, POINTER(c_double), POINTER(c_double)]

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

    # optimize/update model

    GRBoptimize = grblib.GRBoptimize
    GRBoptimize.restype = c_int
    GRBoptimize.argtypes = [c_void_p]

    GRBupdatemodel = grblib.GRBupdatemodel
    GRBupdatemodel.restype = c_int
    GRBupdatemodel.argtypes = [c_void_p]

    # read/write files

    GRBwrite = grblib.GRBwrite
    GRBwrite.restype = c_int
    GRBwrite.argtypes = [c_void_p, c_char_p]

    GRBreadModel = grblib.GRBreadmodel
    GRBreadModel.restype = c_int
    GRBreadModel.argtypes = [c_void_p, c_char_p, c_void_p]

    GRBgetenv = grblib.GRBgetenv
    GRBgetenv.restype = c_void_p
    GRBgetenv.argtypes = [c_void_p]

# vim: ts=4 sw=4 et
