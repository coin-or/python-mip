from mip.model import *
from ctypes import *
from ctypes.util import *
from array import array


class SolverGurobi(Solver):

    def __init__(self, model: Model, name: str, sense: str):
        super().__init__(model, name, sense)

        # setting class members to default values
        self._log = ""
        self._env = c_void_p(0)
        self._model = c_void_p(0)
        self._callback = None

        # creating Gurobi environment
        if GRBloadenv(byref(self._env), c_str(self._log)) != 0:
            raise Exception('Gurobi environment could not be loaded,\
check your license.')

        # creating Gurobi model
        numvars = c_int(0)
        obj = c_double(0)
        lb = c_double(0)
        ub = c_double(0)
        vtype = c_char_p()
        varnames = c_void_p(0)
        if GRBnewmodel(self._env, byref(self._model), c_str(name), numvars,
                       byref(obj), byref(lb), byref(ub), vtype, varnames) != 0:
            raise Exception('Could not create Gurobi model')

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
                var_type: str = CONTINUOUS,
                column: Column = None,
                name: str = ""):
        # collecting column data
        numnz = 0 if column is None else len(column.constrs)
        vind = (c_int * numnz)()
        vval = (c_double * numnz)()

        if not name:
            name = 'x({})'.format(self.num_cols())

        # collecting column coefficients
        for i in range(numnz):
            vind[i] = column.constrs[i].idx
            vval[i] = column.coeffs[i]

        # variable type
        vtype = c_char(ord(var_type))

        error = GRBaddvar(self._model, c_int(numnz), vind, vval, c_double(obj),
                          c_double(lb), c_double(ub),
                          vtype, c_str(name))
        if error != 0:
            raise Exception('Error adding variable {} to model.'.format(name))

        self.__n_cols_buffer += 1
        if vtype == BINARY or vtype == INTEGER:
            self.__n_int_buffer += 1

    def add_constr(self, lin_expr: LinExpr, name: str = ""):
        self.flush_cols()

        # collecting linear expression data
        numnz = len(lin_expr.expr)
        cind = array("i", [var.idx for var in lin_expr.expr.keys()])
        cval = array("d", [coef for coef in lin_expr.expr.values()])

        #cind[:] = [var.idx for var in lin_expr.expr.keys()]
        #cval[:] = [coef for coef in lin_expr.expr.values()]
        # collecting variable coefficients
        #for i, (var, coeff) in enumerate(lin_expr.expr.items()):
        #    cind[i] = var.idx
        #    cval[i] = coeff


        # constraint sense and rhs
        sense = c_char(ord(lin_expr.sense))
        rhs = c_double(-lin_expr.const)

        if not name:
            name = 'r({})'.format(self.num_rows())

        error = GRBaddconstr(self._model, numnz,
                             cast(cind.buffer_info()[0], POINTER(c_int)),
                             cast(cval.buffer_info()[0], POINTER(c_double)),
                             sense,
                             rhs, c_str(name))
        if error != 0:
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
        idxs = list()
        for var in self.model.vars:
            vtype = self.var_get_var_type(var)
            if vtype == BINARY or vtype == INTEGER:
                idxs.append(var.idx)

        ccont = (c_char * len(idxs))()
        for i in range(len(idxs)):
            ccont[i] = CONTINUOUS.encode("utf-8")

        GRBsetcharattrarray(self._model, c_str("VType"), 0, len(idxs), ccont)
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
        rint = min(sys.maxsize, int(rdbl))
        return rint

    def set_max_nodes(self, max_nodes: int):
        self.set_dbl_param("NodeLimit", float(max_nodes))

    def set_num_threads(self, threads: int):
        self.__threads = threads

    def optimize(self) -> OptimizationStatus:
        # todo add branch_selector and incumbent_updater callbacks
        def callback(p_model: c_void_p,
                     p_cbdata: c_void_p,
                     where: int,
                     p_usrdata: c_void_p) -> int:

            # adding cuts
            if self.model.cuts_generator and where == 5:  # MIPNODE == 5
                # obtaining relaxation solution and "translating" it
                cb_solution = (c_double * self.num_cols())()
                GRBcbget(p_cbdata, where, GRB_CB_MIPNODE_REL, cb_solution)
                relax_solution = []
                for i in range(self.num_cols()):
                    if cb_solution[i] <= -EPS or cb_solution[i] >= EPS:
                        relax_solution.append((self.model.vars[i],
                                               cb_solution[i]))

                # calling cuts generator
                cuts = self.model.cuts_generator.generate_cuts(relax_solution)
                # adding cuts
                for lin_expr in cuts:
                    # collecting linear expression data
                    numnz = len(lin_expr.expr)
                    cind = array("i", [var.idx for var in lin_expr.expr.keys()])
                    cval = array("d", [coef for coef in lin_expr.expr.values()])

                    # constraint sense and rhs
                    sense = c_char(ord(lin_expr.sense))
                    rhs = c_double(-lin_expr.const)

                    GRBcbcut(p_cbdata, numnz,
                             cast(cind.buffer_info()[0], POINTER(c_int)),
                             cast(cval.buffer_info()[0], POINTER(c_double)),
                             sense, rhs)

            # adding lazy constraints
            elif self.model.lazy_constrs_generator and where == 4:  # MIPSOL==4
                # obtaining relaxation solution and "translating" it
                cb_solution = (c_double * self.num_cols())()
                GRBcbget(p_cbdata, where, GRB_CB_MIPSOL_SOL, cb_solution)
                solution = []
                for i in range(self.num_cols()):
                    if cb_solution[i] <= -EPS or cb_solution[i] >= EPS:
                        solution.append((self.model.vars[i], cb_solution[i]))

                # calling constraint generator
                lcg = self.model.lazy_constrs_generator
                constrs = lcg.generate_lazy_constrs(solution)
                # adding cuts
                for lin_expr in constrs:
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

                    GRBcblazy(p_cbdata, numnz, cind, cval, sense, rhs)

            return 0

        self.update()
        if self.model.cuts_generator is not None:
            self._callback = GRBcallbacktype(callback)
            GRBsetcallbackfunc(self._model, self._callback, c_void_p(0))

        if self.__threads >= 1:
            self.set_int_param("Threads", self.__threads)

        self.set_int_param("Cuts", self.model.cuts)

        # executing Gurobi to solve the formulation
        status = int(GRBoptimize(self._model))
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

    def var_get_xi(self, var: "Var", i: int) -> float:
        self.set_int_param("SolutionNumber", i)
        return self.get_dbl_attr_element("Xn", var.idx)

    def var_get_index(self, name: str) -> int:
        idx = c_int(0)
        error = GRBgetvarbyname(self._model, c_str(name), byref(idx))
        if error:
            raise Exception("Error calling GRBgetvarbyname")
        return idx.value

    def get_objective_value_i(self, i: int) -> float:
        self.set_int_param("SolutionNumber", i)
        return self.get_dbl_attr("PoolObjVal")

    def get_objective_value(self) -> float:
        return self.get_dbl_attr('ObjVal')

    def set_processing_limits(self,
                              max_time: float = inf,
                              max_nodes: float = inf,
                              max_sol: int = inf):
        # todo: Set limits even when they are 'inf'
        if max_time != inf:
            self.set_dbl_param("TimeLimit", max_time)
        if max_nodes != inf:
            self.set_dbl_param("NodeLimit", max_nodes)
        if max_sol != inf:
            self.set_int_param("SolutionLimit", max_sol)

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
        const = lin_expr.const

        # resetting objective function
        num_vars = c_int(self.num_cols())
        zeros = (c_double * self.num_cols())()
        for i in range(self.num_cols()):
            zeros[i] = 0.0
        error = GRBsetdblattrarray(self._model, c_str("Obj"), c_int(0),
                                   num_vars, zeros)
        if error != 0:
            raise Exception('Could not set gurobi double attribute array Obj')

        # setting objective sense
        if sense == MAXIMIZE:
            self.set_int_attr("ModelSense", -1)
        elif sense == MINIMIZE:
            self.set_int_attr("ModelSense", 1)

        # setting objective function
        self.set_dbl_attr("ObjCon", const)
        error = GRBsetdblattrlist(self._model, c_str("Obj"), c_int(numnz),
                                  cind, cval)
        if error != 0:
            raise Exception("Error modifying attribute Obj")
        self.__n_modified_cols += 1

    def set_objective_const(self, const: float) -> None:
        self.set_dbl_attr("ObjCon", const)
        self.__updated = False

    def set_start(self, start: List[Tuple["Var", float]]) -> None:
        # collecting data
        numnz = len(start)
        cind = (c_int * numnz)()
        cval = (c_double * numnz)()

        # collecting variable coefficients
        for i in range(len(start)):
            cind[i] = start[i][0].idx
            cval[i] = start[i][1]

        error = GRBsetdblattrlist(self._model, c_str("Start"), numnz,
                                  cind, cval)
        if error != 0:
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
        GRBwrite(self._model, c_str(file_path))

    def read(self, file_path: str) -> None:
        GRBfreemodel(self._model)
        self._model = c_void_p(0)
        GRBreadModel(self._env, c_str(file_path), byref(self._model))

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

        numnz = c_int()
        cbeg = POINTER(c_int)()
        cind = POINTER(c_int)()
        cval = POINTER(c_double)()

        # obtaining number of non-zeros
        GRBgetconstrs(self._model, byref(numnz), cbeg, cind, cval,
                      c_int(constr.idx), c_int(1))

        # creating arrays to hold indices and coefficients
        cbeg = (c_int * 2)()  # beginning and ending
        cind = (c_int * numnz.value)()
        cval = (c_double * numnz.value)()

        # obtaining variables and coefficients
        GRBgetconstrs(self._model, byref(numnz), cbeg, cind, cval,
                      c_int(constr.idx), c_int(1))

        # obtaining sense and rhs
        c_sense = c_char()
        rhs = c_double()
        GRBgetcharattrelement(self._model, c_str("Sense"), c_int(constr.idx),
                              byref(c_sense))
        GRBgetdblattrelement(self._model, c_str("RHS"),
                             c_int(constr.idx), byref(rhs))

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
        self.flush_rows()
        return self.get_str_attr_element('ConstrName', idx)

    def constr_set_expr(self, constr: Constr, value: LinExpr) -> LinExpr:
        raise NotImplementedError("Gurobi functionality currently unavailable")

    def constr_get_pi(self, constr: "Constr") -> float:
        return self.get_dbl_attr("Pi", constr.idx)

    def constr_get_index(self, name: str) -> int:
        idx = c_int(0)
        error = GRBgetconstrbyname(self._model, c_str(name), byref(idx))
        if error:
            raise Exception("Error alling GRBgetconstrbyname")
        return idx.value

    def var_get_lb(self, var: "Var") -> float:
        self.flush_cols()
        return self.get_dbl_attr_element("LB", var.idx)

    def var_set_lb(self, var: "Var", value: float) -> None:
        self.set_dbl_attr_element("LB", var.idx, value)
        self.__n_modified_cols += 1

    def var_get_ub(self, var: "Var") -> float:
        self.flush_cols()
        return self.get_dbl_attr_element("UB", var.idx)

    def var_set_ub(self, var: "Var", value: float) -> None:
        self.set_dbl_attr_element("UB", var.idx, value)
        self.__n_modified_cols += 1

    def var_get_obj(self, var: "Var") -> float:
        self.flush_cols()
        return self.get_dbl_attr_element("Obj", var.idx)

    def var_set_obj(self, var: "Var", value: float) -> None:
        self.set_dbl_attr_element("Obj", var.idx, value)
        self.__n_modified_cols += 1

    def var_get_var_type(self, var: "Var") -> str:
        self.flush_cols()
        res = c_char(0)
        GRBgetcharattrelement(self._model, c_str("VType"),
                              c_int(var.idx), byref(res))

        if res.value == b"B":
            return BINARY
        elif res.value == b"C":
            return CONTINUOUS
        elif res.value == b"I":
            return INTEGER

        raise ValueError("Gurobi: invalid variable type returned...")

    def var_set_var_type(self, var: "Var", value: str) -> None:
        self.set_char_attr_element("VType", var.idx, value)
        self._updated = False

    def var_get_column(self, var: "Var"):
        self.update()

        numnz = c_int()
        cbeg = POINTER(c_int)()
        cind = POINTER(c_int)()
        cval = POINTER(c_double)()

        # obtaining number of non-zeros
        error = GRBgetvars(self._model, byref(numnz), cbeg, cind, cval,
                           c_int(var.idx), c_int(1))
        if error != 0:
            raise Exception('Error querying gurobi model information')

        # creating arrays to hold indices and coefficients
        cbeg = (c_int * 2)()  # beginning and ending
        cind = (c_int * numnz.value)()
        cval = (c_double * numnz.value)()

        # obtaining variables and coefficients
        error = GRBgetvars(self._model, byref(numnz), cbeg, cind, cval,
                           c_int(var.idx), c_int(1))
        if error != 0:
            raise Exception('Error querying gurobi model information')

        constr = [self.model.constrs[cind[i]] for i in range(numnz.value)]
        coefs = [float(cval[i]) for i in range(numnz.value)]

        return Column(constr, coefs)

    def var_set_column(self, var: "Var", value: Column):
        raise NotImplementedError("Gurobi functionality currently unavailable")

    def var_get_rc(self, var: "Var") -> float:
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
        #print('GUROBI UPDATE')
        GRBupdatemodel(self._model)
        self.__n_cols_buffer = 0
        self.__n_int_buffer = 0
        self.__n_rows_buffer = 0
        self.__n_modified_cols = 0
        self.__n_modified_rows = 0
        self.__updated = True

    def set_char_attr_element(self, name: str, index: int, value: str):
        assert len(value) == 1
        error = GRBsetcharattrelement(self._model, c_str(name),
                                      c_int(index), c_char(ord(value)))
        if error != 0:
            raise Exception(
                'Error setting gurobi char attr element {} index {} to value'.
                format(name, index, value))

    def get_dbl_attr_element(self, name: str, index: int) -> float:
        res = c_double(0.0)
        error = GRBgetdblattrelement(self._model, c_str(name),
                                     c_int(index), byref(res))
        if error != 0:
            raise Exception('Error get grb double attr element {} index {}'.
                            format(name, index))
        return res.value

    def set_dbl_attr_element(self, name: str, index: int, value: float):
        error = GRBsetdblattrelement(self._model, c_str(name),
                                     c_int(index), c_double(value))
        if error != 0:
            raise Exception(
                "Error modifying dbl attribute {} for element {} to value {}".
                format(name, index, value))

    def set_int_attr(self, name: str, value: int):
        error = GRBsetintattr(self._model, c_str(name), c_int(value))
        if error != 0:
            raise Exception("Error modifying int attribute {} to {}".
                            format(name, value))

    def set_dbl_attr(self, name: str, value: float):
        error = GRBsetdblattr(self._model, c_str(name), c_double(value))
        if error != 0:
            raise Exception("Error modifying double attribute {} to {}".
                            format(name, value))

    def get_int_attr(self, name: str) -> int:
        res = c_int(0)
        error = GRBgetintattr(self._model, c_str(name), byref(res))
        if error != 0:
            raise Exception('Error getting int attribute {}'.format(name))
        return res.value

    def get_int_param(self, name: str) -> int:
        res = c_int(0)
        error = GRBgetintparam(GRBgetenv(self._model), c_str(name), byref(res))
        if error != 0:
            raise Exception("Error getting gurobi integer parameter {}".
                            format(name))
        return res.value

    def set_int_param(self, name: str, value: int):
        error = GRBsetintparam(GRBgetenv(self._model),
                               c_str(name), c_int(value))
        if error != 0:
            raise Exception("Error mofifying int parameter {} to value {}".
                            format(name, value))

    def get_dbl_attr(self, attr: str) -> float:
        res = c_double(0.0)
        error = GRBgetdblattr(self._model, c_str(attr), byref(res))
        if error != 0:
            raise Exception('Error getting gurobi double attribute {}'.
                            format(attr))
        return res.value

    def set_dbl_param(self, param: str, value: float):
        error = GRBsetdblparam(GRBgetenv(self._model), c_str(param),
                               c_double(value))
        if error != 0:
            raise Exception("Error setting gurobi double param " +
                            param + " to {}".format(value))

    def get_dbl_param(self, param: str) -> float:
        res = c_double()
        error = GRBgetdblparam(GRBgetenv(self._model), c_str(param),
                               byref(res))
        if error != 0:
            raise Exception("Error getting gurobi double parameter {}".
                            format(param))
        return res.value

    def get_str_attr_element(self, attr: str, index: int) -> str:
        vName = c_char_p(0)
        error = GRBgetstrattrelement(self._model, c_str(attr), c_int(index),
                                     byref(vName))
        if error != 0:
            raise Exception('Error getting str attribute {} index {}'.
                            format(attr, index))
        return vName.value.decode('utf-8')


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
                libPath = find_library('gurobi{}{}'.format(majorVersion,
                                                           minorVersion))
                if libPath is not None:
                    break
            except:
                continue
        if libPath is not None:
            break

    if libPath is None:
        raise Exception()
    grblib = CDLL(libPath)
    print('gurobi version {}.{} found'.format(majorVersion,
                                              minorVersion))
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
    GRBnewmodel.argtypes = [c_void_p, c_void_p, c_char_p, c_int,
                            POINTER(c_double), POINTER(c_double),
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
    GRBgetdblattrarray.argtypes = [c_void_p, c_char_p, c_int, c_int,
                                   POINTER(c_double)]

    GRBsetdblattrarray = grblib.GRBsetdblattrarray
    GRBsetdblattrarray.restype = c_int
    GRBsetdblattrarray.argtypes = [c_void_p, c_char_p, c_int, c_int,
                                   POINTER(c_double)]

    GRBsetdblattrlist = grblib.GRBsetdblattrlist
    GRBsetdblattrlist.restype = c_int
    GRBsetdblattrlist.argtypes = [c_void_p, c_char_p, c_int, POINTER(c_int),
                                  POINTER(c_double)]

    GRBgetdblattrelement = grblib.GRBgetdblattrelement
    GRBgetdblattrelement.restype = c_int
    GRBgetdblattrelement.argtypes = [c_void_p, c_char_p, c_int,
                                     POINTER(c_double)]

    GRBsetdblattrelement = grblib.GRBsetdblattrelement
    GRBsetdblattrelement.restype = c_int
    GRBsetdblattrelement.argtypes = [c_void_p, c_char_p, c_int, c_double]

    GRBsetcharattrarray = grblib.GRBsetcharattrarray
    GRBsetcharattrarray.restype = c_int
    GRBsetcharattrarray.argtypes = [c_void_p, c_char_p, c_int, c_int, c_char_p]

    GRBgetcharattrelement = grblib.GRBgetcharattrelement
    GRBgetcharattrelement.restype = c_int
    GRBgetcharattrelement.argtypes = [c_void_p, c_char_p, c_int,
                                      POINTER(c_char)]

    GRBsetcharattrelement = grblib.GRBsetcharattrelement
    GRBsetcharattrelement.restype = c_int
    GRBsetcharattrelement.argtypes = [c_void_p, c_char_p, c_int, c_char]

    GRBgetstrattrelement = grblib.GRBgetstrattrelement
    GRBgetstrattrelement.argtypes = [c_void_p, c_char_p, c_int,
                                     POINTER(c_char_p)]
    GRBgetstrattrelement.restype = c_int

    # manipulate parameter(s)

    GRBgetintparam = grblib.GRBgetintparam
    GRBgetintparam.argtypes = [c_void_p, c_char_p,
                               POINTER(c_int)]
    GRBgetintparam.restype = c_int

    GRBsetintparam = grblib.GRBsetintparam
    GRBsetintparam.argtypes = [c_void_p, c_char_p, c_int]
    GRBsetintparam.restype = c_int

    GRBgetdblparam = grblib.GRBgetdblparam
    GRBgetdblparam.argtypes = [c_void_p, c_char_p,
                               POINTER(c_double)]
    GRBgetdblparam.restype = c_int

    GRBsetdblparam = grblib.GRBsetdblparam
    GRBsetdblparam.argtypes = [c_void_p, c_char_p, c_double]
    GRBsetdblparam.restype = c_int

    # manipulate objective function(s)

    GRBsetobjectiven = grblib.GRBsetobjectiven
    GRBsetobjectiven.restype = c_int
    GRBsetobjectiven.argtypes = [c_void_p, c_int, c_int, c_double,
                                 c_double, c_double, c_char_p,
                                 c_double, c_int, POINTER(c_int),
                                 POINTER(c_double)]

    # add variables and constraints

    GRBaddvar = grblib.GRBaddvar
    GRBaddvar.restype = c_int
    GRBaddvar.argtypes = [c_void_p, c_int, POINTER(c_int),
                          POINTER(c_double), c_double, c_double,
                          c_double, c_char, c_char_p]

    GRBaddconstr = grblib.GRBaddconstr
    GRBaddconstr.restype = c_int
    GRBaddconstr.argtypes = [c_void_p, c_int, POINTER(c_int),
                             POINTER(c_double), c_char, c_double,
                             c_char_p]

    # get constraints

    GRBgetconstrs = grblib.GRBgetconstrs
    GRBgetconstrs.restype = c_int
    GRBgetconstrs.argtypes = [c_void_p, POINTER(c_int),
                              POINTER(c_int), POINTER(c_int),
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
    GRBcbcut.argtypes = [c_void_p, c_int, POINTER(c_int), POINTER(c_double),
                         c_char, c_double]

    GRBcbget = grblib.GRBcbget
    GRBcbget.restype = c_int
    GRBcbget.argtypes = [c_void_p, c_int, c_int, c_void_p]

    GRBcblazy = grblib.GRBcblazy
    GRBcblazy.restype = c_int
    GRBcblazy.argtypes = [c_void_p, c_int, POINTER(c_int), POINTER(c_double),
                          c_char, c_double]

    GRBcbsolution = grblib.GRBcbsolution
    GRBcbsolution.restype = c_int
    GRBcbsolution.argtypes = [c_void_p, POINTER(c_double), POINTER(c_double)]

    GRBgetvarbyname = grblib.GRBgetvarbyname
    GRBgetvarbyname.restype = c_int
    GRBgetvarbyname.argtypes = [c_void_p, c_char_p, POINTER(c_int)]

    GRBgetconstrbyname = grblib.GRBgetconstrbyname
    GRBgetconstrbyname.restype = c_int
    GRBgetconstrbyname.argtypes = [c_void_p, c_char_p, POINTER(c_int)]

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
