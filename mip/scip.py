from mip.model import *
from ctypes import *
from ctypes.util import *


class SolverSCIP(Solver):

    def __init__(self, name: str, sense: str):
        super().__init__(name, sense)

        # setting class members to default values
        self._num_vars: int = 0
        self._num_constrs: int = 0
        self._log: str = ""
        self._scip: c_void_p = c_void_p(0)

        # setting variables for an empty model
        numvars: c_int = c_int(0)
        obj: c_double = c_double(0)
        lb: c_double = c_double(0)
        ub: c_double = c_double(0)
        vtype: c_char_p = c_char_p()
        varnames: c_void_p = c_void_p(0)

        # creating Gurobi environment
        if GRBloadenv(byref(self._env), c_str(self._log)) != 0:
            # todo: environment could not be loaded
            pass

        # creating Gurobi model
        if GRBnewmodel(self._env, byref(self._model), c_str(name), numvars, byref(obj), byref(lb), byref(ub), vtype, varnames) != 0:
            # todo: model could not be generated
            pass

    def __del__(self):
        # freeing Gurobi model and environment
        if self._model:
            GRBfreemodel(self._model)
        if self._env:
            GRBfreeenv(self._env)

    def add_var(self, obj: float = 0,
                lb: float = 0,
                ub: float = float("inf"),
                column: "Column" = None,
                type: str = 'C',
                name: str = "") -> int:
        # collecting column data
        numnz: c_int = 0 if column is None else len(column.constrs)
        vind: POINTER(c_int) = (c_int * numnz)()
        vval: POINTER(c_double) = (c_double * numnz)()

        # collecting column coefficients
        for i in range(numnz):
            vind[i] = column.constrs[i].idx
            vval[i] = column.coeffs[i]

        # variable type
        vtype: c_char = c_char(ord(type))

        # variable index
        idx: int = self._num_vars
        self._num_vars += 1

        GRBaddvar(self._model, c_int(numnz), vind, vval, c_double(obj), c_double(lb), c_double(ub), vtype, c_str(name))
        return idx

    def add_constr(self, lin_expr: "LinExpr", name: str = '') -> int:
        # collecting linear expression data
        numnz: c_int = len(lin_expr.expr)
        cind: POINTER(c_int) = (c_int * numnz)()
        cval: POINTER(c_double) = (c_double * numnz)()

        # collecting variable coefficients
        for i, (var, coeff) in enumerate(lin_expr.expr.items()):
            cind[i] = var.idx
            cval[i] = coeff

        # constraint sense and rhs
        sense: c_char = c_char(ord(lin_expr.sense))
        rhs: c_double = c_double(-lin_expr.const)

        # constraint index
        idx: int = self._num_constrs
        self._num_constrs += 1

        GRBaddconstr(self._model, numnz, cind, cval, sense, rhs, c_str(name))
        return idx

    def optimize(self) -> int:
        # executing Gurobi to solve the formulation
        status: int = int(GRBoptimize(self._model))

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

        return status

    def set_objective(self, lin_expr: "LinExpr", sense: str = ""):
        # collecting linear expression data
        numnz: c_int = len(lin_expr.expr)
        cind: POINTER(c_int) = (c_int * numnz)()
        cval: POINTER(c_double) = (c_double * numnz)()

        # collecting variable coefficients
        for i, (var, coeff) in enumerate(lin_expr.expr.items()):
            cind[i] = var.idx
            cval[i] = coeff

        # objective function constant
        const = c_double(lin_expr.const)

        # resetting objective function
        num_vars: c_int = c_int(self._num_vars)
        zeros: POINTER(c_double) = (c_double * self._num_vars)()
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

        # (the function may be used for multi-objective models)
        # index = c_int(0)
        # priority = c_int(1)
        # weight = c_double(0.0)
        # abstol = c_double(0.0)
        # reltol = c_double(0.0)
        # name = c_str("primary")
        # const = c_double(lin_expr.const)
        # GRBsetobjectiven(self._model, index, priority, weight, abstol,
        #                  reltol, name, const, lnz, lind, lval)

        return True

    def set_start(self, variables: List["Var"], values: List[float]):
        # collecting data
        numnz: c_int = len(variables)
        cind: POINTER(c_int) = (c_int * numnz)()
        cval: POINTER(c_double) = (c_double * numnz)()

        # collecting variable coefficients
        for i in range(len(variables)):
            cind[i] = variables[i].idx
            cval[i] = values[i]

        GRBsetdblattrlist(self._model, "Start", numnz, cind, cval)

    def write(self, file_path: str):
        # writing formulation to output file
        GRBwrite(self._model, c_str(file_path))


# auxiliary functions
def c_str(value) -> c_char_p:
    """
    This function converts a python string into a C compatible char[]
    :param value: input string
    :return: string converted to C's format
    """
    return create_string_buffer(value.encode('utf-8'))


grblib = CDLL(find_library('gurobi80'))

# create/release environment and model

GRBloadenv = grblib.GRBloadenv
GRBloadenv.restype = c_int
GRBloadenv.argtypes = [c_void_p, c_char_p]

GRBnewmodel = grblib.GRBnewmodel
GRBnewmodel.restype = c_int
GRBnewmodel.argtypes = [c_void_p, c_void_p, c_char_p, c_int, POINTER(c_double),
                        POINTER(c_double), POINTER(c_double), c_char_p, c_void_p]

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

GRBgetdblattr = grblib.GRBgetdblattr
GRBgetdblattr.restype = c_int
GRBgetdblattr.argtypes = [c_void_p, c_char_p, POINTER(c_double)]

GRBsetdblattr = grblib.GRBsetdblattr
GRBsetdblattr.restype = c_int
GRBsetdblattr.argtypes = [c_void_p, c_char_p, c_double]

GRBsetdblattrarray = grblib.GRBsetdblattrarray
GRBsetdblattrarray.restype = c_int
GRBsetdblattrarray.argtypes = [c_void_p, c_char_p, c_int, c_int, POINTER(c_double)]

GRBsetdblattrlist = grblib.GRBsetdblattrlist
GRBsetdblattrlist.restype = c_int
GRBsetdblattrlist.argtypes = [c_void_p, c_char_p, c_int, POINTER(c_int), POINTER(c_double)]

# manipulating objective function(s)
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

# optimize

GRBoptimize = grblib.GRBoptimize
GRBoptimize.restype = c_int
GRBoptimize.argtypes = [c_void_p]

# read/write files

GRBwrite = grblib.GRBwrite
GRBwrite.restype = c_int
GRBwrite.argtypes = [c_void_p, c_char_p]
