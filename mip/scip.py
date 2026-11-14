"""Python-MIP interface to SCIP through the optional PySCIPOpt package."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from numbers import Real

import mip

try:
    import pyscipopt as scip
except ModuleNotFoundError as error:
    if error.name != "pyscipopt":
        raise
    scip = None

has_scip = scip is not None
logger = logging.getLogger(__name__)


@dataclass
class _Variable:
    native: scip.Variable
    name: str
    lb: mip.Numeric
    ub: mip.Numeric
    obj: mip.Numeric
    var_type: str
    priority: int = 0


@dataclass
class _Row:
    native: scip.Constraint
    name: str
    rhs: mip.Numeric
    sense: str
    coefficients: dict[int, mip.Numeric] = field(default_factory=dict)


class SolverScip(mip.Solver):
    def __init__(self, model: mip.Model, name: str, sense: str):
        if not has_scip:
            raise FileNotFoundError(
                "SCIP not found. Install PySCIPOpt with `pip install mip[scip]`."
            )
        super().__init__(model, name, sense)
        self._scip = scip.Model(name)
        self._scip.setIntParam("parallel/maxnthreads", 1)
        self._vars: list[_Variable] = []
        self._rows: list[_Row] = []
        self._var_names: dict[str, int] = {}
        self._row_names: dict[str, int] = {}
        self._sense = sense
        self._objective_const: mip.Numeric = 0
        self._emphasis = mip.SearchEmphasis.DEFAULT
        self._cutoff = mip.INF
        self._verbose = 1
        self._start: list[tuple[mip.Var, mip.Numeric]] = []
        self._lp_params: dict[str, object] | None = None
        self._clear_solution()
        self.set_objective_sense(sense)

    def _clear_solution(self):
        self._status = mip.OptimizationStatus.LOADED
        self._solutions: list[list[float]] = []
        self._objectives: list[float] = []
        self._bound: float | None = None
        self._pi: list[float | None] = []
        self._rc: list[float | None] = []

    def _prepare(self):
        if self._scip.getStage() > scip.SCIP_STAGE.PROBLEM:
            self._scip.freeTransform()
        self._clear_solution()

    @staticmethod
    def _number(value):
        if not isinstance(value, Real):
            raise TypeError("SCIP model coefficients and bounds must be real numbers")
        if math.isnan(value):
            raise ValueError("SCIP model coefficients and bounds cannot be NaN")
        return value

    def _infinity(self, value: mip.Numeric) -> float:
        if value == mip.INF:
            return self._scip.infinity()
        if value == -mip.INF:
            return -self._scip.infinity()
        return float(value)

    def _from_native_bound(self, value: float) -> float:
        if self._scip.isInfinity(abs(value)):
            return math.copysign(mip.INF, value)
        return value

    @staticmethod
    def _bounds(lb, ub, var_type):
        if var_type != mip.CONTINUOUS:
            lb = math.ceil(lb) if math.isfinite(lb) else lb
            ub = math.floor(ub) if math.isfinite(ub) else ub
        # SCIP rejects contradictory bounds during variable creation. Keep the
        # requested bounds in metadata and report infeasibility on optimize.
        return (0, 0) if lb > ub else (lb, ub)

    def _sync_bounds(self):
        for var in self._vars:
            self._sync_var_bounds(var)

    def _sync_var_bounds(self, var):
        lb, ub = self._bounds(var.lb, var.ub, var.var_type)
        self._scip.chgVarLb(var.native, self._infinity(lb))
        self._scip.chgVarUb(var.native, self._infinity(ub))

    def _infeasible_bounds(self, is_lp):
        return any(
            var.lb > var.ub
            or (
                not is_lp
                and var.var_type != mip.CONTINUOUS
                and math.isfinite(var.lb)
                and math.isfinite(var.ub)
                and math.ceil(var.lb) > math.floor(var.ub)
            )
            for var in self._vars
        )

    def _coefficients(self, expr: mip.LinExpr) -> dict[int, mip.Numeric]:
        coefficients = {}
        for var, coefficient in expr.expr.items():
            if var.model is not self.model or not 0 <= var.idx < len(self._vars):
                raise ValueError("SCIP expressions must use variables from this model")
            self._number(coefficient)
            if coefficient:
                coefficients[var.idx] = coefficient
        return coefficients

    def _create_row(self, expr: mip.LinExpr, name: str) -> _Row:
        coefficients = self._coefficients(expr)
        self._number(expr.const)
        return self._create_row_data(coefficients, -expr.const, expr.sense, name)

    def _create_row_data(
        self, coefficients, rhs, sense, name, native_model=None, variables=None
    ):
        native_model = self._scip if native_model is None else native_model
        variables = self._vars if variables is None else variables
        empty = scip.Expr()
        if sense == mip.LESS_OR_EQUAL:
            constraint = empty <= self._infinity(rhs)
        elif sense == mip.GREATER_OR_EQUAL:
            constraint = empty >= self._infinity(rhs)
        elif sense == mip.EQUAL:
            constraint = empty == self._infinity(rhs)
        else:
            raise mip.InvalidLinExpr(f"Invalid constraint sense: {sense}")
        native = native_model.addCons(constraint, name=name)
        for idx, coefficient in coefficients.items():
            native_model.addCoefLinear(native, variables[idx].native, coefficient)
        return _Row(native, name, rhs, sense, coefficients)

    def _reindex(self):
        self._var_names = {}
        self._row_names = {}
        for idx, var in enumerate(self._vars):
            self._var_names.setdefault(var.name, idx)
        for idx, row in enumerate(self._rows):
            self._row_names.setdefault(row.name, idx)

    def add_var(
        self, obj=0, lb=0, ub=mip.INF, var_type=mip.CONTINUOUS, column=None, name=""
    ):
        self._number(obj)
        self._number(lb)
        self._number(ub)
        if var_type not in (mip.CONTINUOUS, mip.INTEGER, mip.BINARY):
            raise ValueError(f"Invalid variable type: {var_type}")
        if column is not None:
            if len(column.constrs) != len(column.coeffs):
                raise ValueError("Column constraints and coefficients must match")
            for cons in column.constrs:
                if (
                    not 0 <= cons.idx < len(self._rows)
                    or self.model.constrs[cons.idx] is not cons
                ):
                    raise ValueError("Column constraints must belong to this model")
            for coefficient in column.coeffs:
                self._number(coefficient)
        self._prepare()
        idx = len(self._vars)
        name = name or f"var({idx})"
        native_lb, native_ub = self._bounds(lb, ub, var_type)
        native = self._scip.addVar(
            name=name,
            vtype=var_type,
            obj=obj,
            lb=self._infinity(native_lb),
            ub=self._infinity(native_ub),
        )
        self._vars.append(_Variable(native, name, lb, ub, obj, var_type))
        self._var_names.setdefault(name, idx)
        if column is not None:
            for cons, coefficient in zip(column.constrs, column.coeffs):
                row = self._rows[cons.idx]
                self._scip.addCoefLinear(row.native, native, coefficient)
                row.coefficients[idx] = row.coefficients.get(idx, 0) + coefficient
                if not row.coefficients[idx]:
                    del row.coefficients[idx]

    def add_constr(self, lin_expr, name=""):
        self._prepare()
        row = self._create_row(lin_expr, name)
        self._row_names.setdefault(name, len(self._rows))
        self._rows.append(row)

    def add_lazy_constr(self, lin_expr):
        raise NotImplementedError("SCIP lazy-constraint callbacks are not implemented")

    def add_cut(self, lin_expr):
        raise NotImplementedError("SCIP cut callbacks are not implemented")

    def add_sos(self, sos, sos_type):
        raise NotImplementedError("SCIP SOS constraints are not implemented")

    def generate_cuts(self, *args, **kwargs):
        raise NotImplementedError("SCIP manual cut generation is not implemented")

    def clique_merge(self, constrs=None):
        raise NotImplementedError("SCIP clique merging is not implemented")

    def _set_lp_mode(self, enabled: bool):
        if enabled and self._lp_params is None:
            keys = {
                "presolving/maxrounds",
                "propagating/maxrounds",
                "propagating/maxroundsroot",
                "limits/maxorigsol",
                "misc/transorigsols",
            }
            self._lp_params = {
                key: value
                for key, value in self._scip.getParams().items()
                if key in keys or key.startswith("heuristics/")
            }
            self._scip.setPresolve(scip.SCIP_PARAMSETTING.OFF)
            self._scip.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
            self._scip.disablePropagation()
            # Original incumbents can cause SCIP to discard LP dual values on
            # re-solves. LP starts must not bypass the root simplex solve.
            self._scip.setIntParam("limits/maxorigsol", 0)
            self._scip.setBoolParam("misc/transorigsols", False)
        elif not enabled and self._lp_params is not None:
            for key, value in self._lp_params.items():
                self._scip.setParam(key, value)
            self._lp_params = None

    def optimize(self, relax=False, lp_preprocess=False):
        if self.model.lazy_constrs_generator or self.model.cuts_generator:
            raise NotImplementedError("SCIP Python-MIP callbacks are not implemented")
        if self.model.store_search_progress_log:
            raise NotImplementedError("SCIP search progress logging is not implemented")
        algorithms = {
            mip.LP_Method.AUTO: "s",
            mip.LP_Method.PRIMAL: "p",
            mip.LP_Method.DUAL: "d",
        }
        if self.model.lp_method not in algorithms:
            raise NotImplementedError(f"SCIP does not support {self.model.lp_method}")
        is_lp = relax or self.num_int() == 0
        self._set_lp_mode(is_lp)
        self._prepare()
        # Singleton LP rows must remain rows, not turn into variable bounds:
        # column generation needs their dual multipliers.
        for row in self._rows:
            self._scip.setModifiable(row.native, is_lp)
        if self._infeasible_bounds(is_lp):
            self._status = mip.OptimizationStatus.INFEASIBLE
            return self._status
        original_types = [var.var_type for var in self._vars]
        original_node_limit = self._scip.getParam("limits/nodes")
        try:
            if is_lp and self.get_max_iter() != mip.INT_MAX:
                # Do not retry an iteration-limited LP at new nodes.
                self._scip.setLongintParam(
                    "limits/nodes",
                    min(original_node_limit, 1) if original_node_limit >= 0 else 1,
                )
            if relax:
                for var in self._vars:
                    self._scip.chgVarType(var.native, mip.CONTINUOUS)
                    self._scip.chgVarLb(var.native, self._infinity(var.lb))
                    self._scip.chgVarUb(var.native, self._infinity(var.ub))
            else:
                self._sync_bounds()
            self._scip.setCharParam("lp/initalgorithm", algorithms[self.model.lp_method])
            self._scip.setCharParam(
                "lp/resolvealgorithm", algorithms[self.model.lp_method]
            )
            self._scip.setRealParam(
                "numerics/feastol", min(self.model.infeas_tol, self.model.integer_tol)
            )
            self._scip.setRealParam("numerics/dualfeastol", self.model.opt_tol)
            self._scip.setIntParam("randomization/randomseedshift", self.model.seed)
            self.set_mip_gap(self.model.max_mip_gap)
            self.set_mip_gap_abs(self.model.max_mip_gap_abs)
            self._scip.setIntParam("limits/maxsol", self.model.sol_pool_size)
            if not is_lp:
                self._scip.setIntParam(
                    "presolving/maxrounds", 0 if self.model.preprocess == 0 else -1
                )
                self._scip.setSeparating(
                    scip.SCIP_PARAMSETTING.OFF
                    if self.model.cuts == 0
                    else scip.SCIP_PARAMSETTING.DEFAULT
                )
                if self._start:
                    values = {}
                    for var, value in self._start:
                        if var.model is not self.model or not 0 <= var.idx < len(
                            self._vars
                        ):
                            raise ValueError(
                                "MIP start variables must belong to this model"
                            )
                        values[var.idx] = value
                    if len(values) == len(self._vars):
                        start = self._scip.createOrigSol()
                    else:
                        # Sparse starts may specify only nonzero decisions.
                        self._scip.setRealParam(
                            "heuristics/completesol/maxunknownrate", 1.0
                        )
                        start = self._scip.createPartialSol()
                    for index, value in values.items():
                        self._scip.setSolVal(start, self._vars[index].native, value)
                    if not self._scip.addSol(start):
                        logger.warning("SCIP did not store the supplied MIP start")
            self._scip.optimize()
            self._status = self._optimization_status(is_lp)
            self._bound = self._from_native_bound(self._scip.getDualbound())
            if (
                self._status
                in (
                    mip.OptimizationStatus.OPTIMAL,
                    mip.OptimizationStatus.FEASIBLE,
                    mip.OptimizationStatus.TRUNCATED,
                )
                and self._scip.getNSols()
            ):
                sols = sorted(
                    self._scip.getSols(),
                    key=self._scip.getSolObjVal,
                    reverse=self._sense == mip.MAXIMIZE,
                )
                for sol in sols:
                    self._solutions.append(
                        [self._scip.getSolVal(sol, var.native) for var in self._vars]
                    )
                    self._objectives.append(self._scip.getSolObjVal(sol))
                if is_lp and self._status == mip.OptimizationStatus.OPTIMAL:
                    self._bound = self._objectives[0]
                    sign = -1 if self._sense == mip.MAXIMIZE else 1
                    for row in self._rows:
                        dual = self._scip.getDualsolLinear(row.native)
                        if self._scip.isInfinity(abs(dual)):
                            logger.warning("SCIP LP dual unavailable for %s", row.name)
                            self._pi.append(None)
                        else:
                            self._pi.append(sign * dual)
                    self._rc = [float(var.obj) for var in self._vars]
                    for row, dual in zip(self._rows, self._pi):
                        for idx, coefficient in row.coefficients.items():
                            self._rc[idx] = (
                                self._rc[idx] - coefficient * dual
                                if dual is not None and self._rc[idx] is not None
                                else None
                            )
        finally:
            self._scip.setLongintParam("limits/nodes", original_node_limit)
            if relax:
                self._scip.freeTransform()
                for var, var_type in zip(self._vars, original_types):
                    self._scip.chgVarType(var.native, var_type)
                self._sync_bounds()
        return self._status

    def _optimization_status(self, is_lp):
        status = self._scip.getStatus()
        mapping = {
            "unknown": mip.OptimizationStatus.LOADED,
            "optimal": mip.OptimizationStatus.OPTIMAL,
            "infeasible": mip.OptimizationStatus.INFEASIBLE,
            "unbounded": mip.OptimizationStatus.UNBOUNDED,
            "inforunbd": mip.OptimizationStatus.INF_OR_UNBD,
        }
        if status in mapping:
            return mapping[status]
        if status in (
            "timelimit",
            "nodelimit",
            "totalnodelimit",
            "stallnodelimit",
            "sollimit",
            "bestsollimit",
            "gaplimit",
            "memlimit",
            "userinterrupt",
            "restartlimit",
            "primallimit",
            "duallimit",
        ):
            if is_lp:
                return mip.OptimizationStatus.TRUNCATED
            return (
                mip.OptimizationStatus.FEASIBLE
                if self._scip.getNSols()
                else mip.OptimizationStatus.NO_SOLUTION_FOUND
            )
        raise mip.InterfacingError(f"Unknown SCIP optimization status: {status}")

    def get_status(self):
        return self._status

    def get_objective_value(self):
        return self._objectives[0] if self._objectives else None

    def get_objective_bound(self):
        return self._bound

    def get_num_solutions(self):
        return len(self._solutions)

    def get_objective_value_i(self, i):
        return self._objectives[i]

    def get_objective(self):
        expr = mip.LinExpr(const=self._objective_const, sense=self._sense)
        for idx, var in enumerate(self._vars):
            if var.obj:
                expr.add_var(self.model.vars[idx], var.obj)
        return expr

    def get_objective_const(self):
        return self._objective_const

    def get_objective_sense(self):
        return self._sense

    def set_objective_sense(self, sense):
        if sense not in (mip.MINIMIZE, mip.MAXIMIZE):
            raise ValueError(f"Invalid objective sense: {sense}")
        self._prepare()
        self._sense = sense
        if sense == mip.MAXIMIZE:
            self._scip.setMaximize()
        else:
            self._scip.setMinimize()
        self._apply_cutoff()

    def _apply_cutoff(self):
        limit = (
            -mip.INF
            if self._cutoff == mip.INF and self._sense == mip.MAXIMIZE
            else self._cutoff
        )
        self._scip.setObjlimit(self._infinity(limit))

    def _apply_objective(self):
        expr = scip.quicksum(var.obj * var.native for var in self._vars if var.obj)
        self._scip.setObjective(
            expr + self._objective_const,
            sense="maximize" if self._sense == mip.MAXIMIZE else "minimize",
        )
        self._apply_cutoff()

    def set_objective(self, lin_expr, sense=""):
        coefficients = self._coefficients(lin_expr)
        new_sense = lin_expr.sense or sense or self._sense
        if new_sense not in (mip.MINIMIZE, mip.MAXIMIZE):
            raise ValueError(f"Invalid objective sense: {new_sense}")
        self._prepare()
        for idx, var in enumerate(self._vars):
            var.obj = coefficients.get(idx, 0)
        self._objective_const = lin_expr.const
        self._sense = new_sense
        self._apply_objective()

    def set_objective_const(self, const):
        self._number(const)
        self._prepare()
        self._scip.addObjoffset(const - self._objective_const)
        self._objective_const = const
        self._apply_cutoff()

    def relax(self):
        self._prepare()
        for var in self._vars:
            self._scip.chgVarType(var.native, mip.CONTINUOUS)
            var.var_type = mip.CONTINUOUS
        self._sync_bounds()

    def set_start(self, start):
        values = list(start)
        for var, value in values:
            if var.model is not self.model or not 0 <= var.idx < len(self._vars):
                raise ValueError("MIP start variables must belong to this model")
            self._number(value)
        self._start = values

    def constr_get_expr(self, constr):
        row = self._rows[constr.idx]
        return mip.LinExpr(
            [self.model.vars[idx] for idx in row.coefficients],
            list(row.coefficients.values()),
            const=-row.rhs,
            sense=row.sense,
        )

    def constr_set_expr(self, constr, value):
        self._prepare()
        old = self._rows[constr.idx]
        new = self._create_row(value, old.name)
        self._scip.delCons(old.native)
        self._rows[constr.idx] = new

    def constr_get_rhs(self, idx):
        return self._rows[idx].rhs

    def constr_set_rhs(self, idx, rhs):
        self._number(rhs)
        self._prepare()
        row = self._rows[idx]
        if row.sense in (mip.GREATER_OR_EQUAL, mip.EQUAL):
            self._scip.chgLhs(row.native, self._infinity(rhs))
        if row.sense in (mip.LESS_OR_EQUAL, mip.EQUAL):
            self._scip.chgRhs(row.native, self._infinity(rhs))
        row.rhs = rhs

    def constr_get_name(self, idx):
        return self._rows[idx].name

    def constr_get_index(self, name):
        return self._row_names.get(name, -1)

    def constr_get_pi(self, constr):
        return self._pi[constr.idx] if self._pi else None

    def constr_get_slack(self, constr):
        if not self._solutions:
            return None
        row = self._rows[constr.idx]
        activity = sum(
            coef * self._solutions[0][idx] for idx, coef in row.coefficients.items()
        )
        residual = row.rhs - activity
        return -residual if row.sense == mip.GREATER_OR_EQUAL else residual

    def remove_constrs(self, constrsList):
        self._prepare()
        removed = set(constrsList)
        for idx in sorted(removed):
            self._scip.delCons(self._rows[idx].native)
        self._rows = [row for idx, row in enumerate(self._rows) if idx not in removed]
        self._reindex()

    def var_get_lb(self, var):
        return self._vars[var.idx].lb

    def var_get_ub(self, var):
        return self._vars[var.idx].ub

    def var_set_lb(self, var, value):
        self._number(value)
        self._prepare()
        self._vars[var.idx].lb = value
        self._sync_var_bounds(self._vars[var.idx])

    def var_set_ub(self, var, value):
        self._number(value)
        self._prepare()
        self._vars[var.idx].ub = value
        self._sync_var_bounds(self._vars[var.idx])

    def var_get_obj(self, var):
        return self._vars[var.idx].obj

    def var_set_obj(self, var, value):
        self._number(value)
        self._prepare()
        self._vars[var.idx].obj = value
        self._apply_objective()

    def var_get_var_type(self, var):
        return self._vars[var.idx].var_type

    def var_set_var_type(self, var, value):
        if value not in (mip.CONTINUOUS, mip.INTEGER, mip.BINARY):
            raise ValueError(f"Invalid variable type: {value}")
        self._prepare()
        data = self._vars[var.idx]
        if value == mip.BINARY and (data.lb < 0 or data.ub > 1):
            data.lb = max(0, data.lb)
            data.ub = min(1, data.ub)
        self._scip.chgVarType(data.native, value)
        data.var_type = value
        self._sync_var_bounds(data)

    def var_get_branch_priority(self, var):
        return self._vars[var.idx].priority

    def var_set_branch_priority(self, var, value):
        if not isinstance(value, int):
            raise ValueError("SCIP branching priority must be an integer")
        self._prepare()
        self._scip.chgVarBranchPriority(self._vars[var.idx].native, value)
        self._vars[var.idx].priority = value

    def var_get_column(self, var):
        constrs, coeffs = [], []
        for idx, row in enumerate(self._rows):
            if var.idx in row.coefficients:
                constrs.append(self.model.constrs[idx])
                coeffs.append(row.coefficients[var.idx])
        return mip.Column(constrs, coeffs)

    def var_set_column(self, var, value):
        raise NotImplementedError("SCIP column replacement is not implemented")

    def var_get_x(self, var):
        return self._solutions[0][var.idx] if self._solutions else None

    def var_get_xi(self, var, i):
        return self._solutions[i][var.idx]

    def var_get_rc(self, var):
        return self._rc[var.idx] if self._rc else None

    def var_get_name(self, idx):
        return self._vars[idx].name

    def var_get_index(self, name):
        return self._var_names.get(name, -1)

    def remove_vars(self, varsList):
        self._prepare()
        removed = set(varsList)
        remap = {}
        for idx, var in enumerate(self._vars):
            if idx not in removed:
                remap[idx] = len(remap)
        for row in self._rows:
            for idx in removed & row.coefficients.keys():
                self._scip.delCoefLinear(row.native, self._vars[idx].native)
            row.coefficients = {
                remap[idx]: coef
                for idx, coef in row.coefficients.items()
                if idx not in removed
            }
        for idx in sorted(removed):
            if not self._scip.delVar(self._vars[idx].native):
                raise mip.InterfacingError(
                    "SCIP could not delete the requested variable"
                )
        self._vars = [var for idx, var in enumerate(self._vars) if idx not in removed]
        self._reindex()

    def num_cols(self):
        return len(self._vars)

    def num_rows(self):
        return len(self._rows)

    def num_nz(self):
        return sum(len(row.coefficients) for row in self._rows)

    def num_int(self):
        return sum(var.var_type != mip.CONTINUOUS for var in self._vars)

    def get_problem_name(self):
        return self._scip.getProbName()

    def set_problem_name(self, name):
        self._scip.setProbName(name)

    def set_processing_limits(
        self,
        max_time=mip.INF,
        max_nodes=mip.INT_MAX,
        max_sol=mip.INT_MAX,
        max_seconds_same_incumbent=mip.INF,
        max_nodes_same_incumbent=mip.INT_MAX,
    ):
        if max_seconds_same_incumbent != mip.INF:
            raise NotImplementedError("SCIP incumbent time limits are not implemented")
        if max_time != mip.INF:
            self.set_max_seconds(max_time)
        if max_nodes != mip.INT_MAX:
            self.set_max_nodes(max_nodes)
        if max_sol != mip.INT_MAX:
            self.set_max_solutions(max_sol)
        if max_nodes_same_incumbent != mip.INT_MAX:
            self._scip.setLongintParam("limits/stallnodes", max_nodes_same_incumbent)

    def get_max_seconds(self):
        return self._from_native_bound(self._scip.getParam("limits/time"))

    def set_max_seconds(self, max_seconds):
        self._scip.setRealParam("limits/time", self._infinity(max_seconds))

    def get_max_nodes(self):
        value = self._scip.getParam("limits/nodes")
        return mip.INT_MAX if value < 0 else value

    def set_max_nodes(self, max_nodes):
        self._scip.setLongintParam(
            "limits/nodes", -1 if max_nodes == mip.INT_MAX else max_nodes
        )

    def get_max_solutions(self):
        value = self._scip.getParam("limits/solutions")
        return mip.INT_MAX if value < 0 else value

    def set_max_solutions(self, max_solutions):
        self._scip.setIntParam(
            "limits/solutions", -1 if max_solutions == mip.INT_MAX else max_solutions
        )

    def get_max_iter(self):
        value = self._scip.getParam("lp/iterlim")
        return mip.INT_MAX if value < 0 else value

    def set_max_iter(self, max_iter):
        value = -1 if max_iter == mip.INT_MAX else max_iter
        self._scip.setLongintParam("lp/iterlim", value)
        self._scip.setLongintParam("lp/rootiterlim", value)

    def set_num_threads(self, threads):
        if threads < 0:
            raise ValueError("SCIP thread count must be nonnegative")
        self._scip.setIntParam("parallel/maxnthreads", threads or 1)
        self._scip.setIntParam("lp/threads", threads)

    def get_verbose(self):
        return self._verbose

    def set_verbose(self, verbose):
        self._scip.hideOutput(not verbose)
        self._verbose = int(bool(verbose))

    def get_mip_gap(self):
        return self._scip.getParam("limits/gap")

    def set_mip_gap(self, mip_gap):
        self._scip.setRealParam("limits/gap", mip_gap)

    def get_mip_gap_abs(self):
        return self._scip.getParam("limits/absgap")

    def set_mip_gap_abs(self, mip_gap_abs):
        self._scip.setRealParam("limits/absgap", mip_gap_abs)

    def get_cutoff(self):
        return self._cutoff

    def set_cutoff(self, cutoff):
        self._number(cutoff)
        self._prepare()
        self._cutoff = cutoff
        self._apply_cutoff()

    def get_emphasis(self):
        return self._emphasis

    def set_emphasis(self, emph):
        choices = {
            mip.SearchEmphasis.DEFAULT: scip.SCIP_PARAMEMPHASIS.DEFAULT,
            mip.SearchEmphasis.FEASIBILITY: scip.SCIP_PARAMEMPHASIS.FEASIBILITY,
            mip.SearchEmphasis.OPTIMALITY: scip.SCIP_PARAMEMPHASIS.OPTIMALITY,
        }
        self._set_lp_mode(False)
        self._scip.setEmphasis(choices[emph], quiet=True)
        self._emphasis = emph

    def get_pump_passes(self):
        if self._lp_params is not None:
            return self._lp_params["heuristics/feaspump/maxloops"]
        return self._scip.getParam("heuristics/feaspump/maxloops")

    def set_pump_passes(self, passes):
        self._set_lp_mode(False)
        self._scip.setIntParam("heuristics/feaspump/maxloops", passes)
        self._scip.setIntParam("heuristics/feaspump/freq", 0 if passes else -1)

    def reset(self):
        self._prepare()

    def write(self, file_path):
        if file_path.lower().endswith(".bas"):
            raise NotImplementedError("SCIP basis files are not implemented")
        if self._infeasible_bounds(True):
            raise mip.InvalidParameter(
                "Cannot export a SCIP model with contradictory variable bounds"
            )
        # Serialize requested bounds, not SCIP's rounded/presolved bounds, so
        # reading the file and relaxing it preserves the original LP.
        exported = scip.Model(self.get_problem_name())
        exported.hideOutput()
        variables = [
            _Variable(
                exported.addVar(
                    name=var.name,
                    vtype=var.var_type,
                    obj=var.obj,
                    lb=self._infinity(var.lb),
                    ub=self._infinity(var.ub),
                ),
                var.name,
                var.lb,
                var.ub,
                var.obj,
                var.var_type,
            )
            for var in self._vars
        ]
        for row in self._rows:
            self._create_row_data(
                row.coefficients,
                row.rhs,
                row.sense,
                row.name,
                native_model=exported,
                variables=variables,
            )
        exported.addObjoffset(self._objective_const)
        if self._sense == mip.MAXIMIZE:
            exported.setMaximize()
        exported.writeProblem(file_path, trans=False, verbose=False)

    def read(self, file_path):
        if file_path.lower().endswith(".bas"):
            raise NotImplementedError("SCIP basis files are not implemented")
        loaded = scip.Model()
        params = self._scip.getParams()
        if self._lp_params is not None:
            params.update(self._lp_params)
        loaded.setParams(params)
        loaded.hideOutput(not self._verbose)
        loaded.readProblem(file_path)
        native_vars = loaded.getVars(transformed=False)
        indices = {var.ptr(): idx for idx, var in enumerate(native_vars)}
        types = {
            "CONTINUOUS": mip.CONTINUOUS,
            "INTEGER": mip.INTEGER,
            "BINARY": mip.BINARY,
            "IMPLINT": mip.INTEGER,
        }
        variables = [
            _Variable(
                var,
                var.name,
                self._from_native_bound(var.getLbOriginal()),
                self._from_native_bound(var.getUbOriginal()),
                var.getObj(),
                types[var.vtype()],
            )
            for var in native_vars
        ]
        rows = []
        ranged = []
        for cons in loaded.getConss(transformed=False):
            if not cons.isLinear():
                raise NotImplementedError(
                    "SCIP model files must contain only linear constraints"
                )
            coefficients = {
                indices[var.ptr()]: coef
                for var, coef in zip(loaded.getConsVars(cons), loaded.getConsVals(cons))
                if coef
            }
            lhs, rhs = loaded.getLhs(cons), loaded.getRhs(cons)
            if lhs == rhs:
                rows.append(_Row(cons, cons.name, rhs, mip.EQUAL, coefficients))
            elif loaded.isInfinity(-lhs):
                rows.append(
                    _Row(
                        cons,
                        cons.name,
                        self._from_native_bound(rhs),
                        mip.LESS_OR_EQUAL,
                        coefficients,
                    )
                )
            elif loaded.isInfinity(rhs):
                rows.append(
                    _Row(cons, cons.name, lhs, mip.GREATER_OR_EQUAL, coefficients)
                )
            else:
                ranged.append((cons, cons.name, lhs, rhs, coefficients))
        self._scip = loaded
        self._vars = variables
        self._rows = rows
        self._lp_params = None
        self._start = []
        self._sense = (
            mip.MAXIMIZE if loaded.getObjectiveSense() == "maximize" else mip.MINIMIZE
        )
        self._objective_const = loaded.getObjoffset()
        for cons, name, lhs, rhs, coefficients in ranged:
            loaded.delCons(cons)
            self._rows.append(
                self._create_row_data(coefficients, lhs, mip.GREATER_OR_EQUAL, name)
            )
            self._rows.append(
                self._create_row_data(
                    coefficients.copy(), rhs, mip.LESS_OR_EQUAL, f"{name}_upper"
                )
            )
        self._reindex()
        self._clear_solution()
        self._apply_cutoff()

    def cgraph_density(self):
        raise NotImplementedError("SCIP conflict graph access is not implemented")

    def conflicting(self, e1, e2):
        raise NotImplementedError("SCIP conflict graph access is not implemented")

    def conflicting_nodes(self, v1):
        raise NotImplementedError("SCIP conflict graph access is not implemented")

    def feature_values(self):
        raise NotImplementedError("SCIP feature extraction is not implemented")

    def feature_names(self):
        raise NotImplementedError("SCIP feature extraction is not implemented")
