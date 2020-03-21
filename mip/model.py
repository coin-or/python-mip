from os import environ
from os.path import isfile
from sys import stdout as out
from typing import List, Tuple, Optional, Union, Dict, Any
from mip.constants import (
    INF,
    MINIMIZE,
    MAXIMIZE,
    GUROBI,
    CBC,
    CONTINUOUS,
    LP_Method,
    OptimizationStatus,
    SearchEmphasis,
    VERSION,
    BINARY,
    INTEGER,
    CutType,
)
from mip.callbacks import ConstrsGenerator, CutPool
from mip.log import ProgressLog
from mip.lists import ConstrList, VarList
from mip.entities import Column, Constr, LinExpr, Var
from mip.exceptions import (
    InvalidLinExpr,
    InfeasibleSolution,
    SolutionNotAvailable,
)
from mip.solver import Solver


class Model:
    """ Mixed Integer Programming Model

    This is the main class, providing methods for building, optimizing,
    querying optimization results and re-optimizing Mixed-Integer Programming
    Models.

    To check how models are created please see the
    :ref:`examples <chapExamples>` included.

    Attributes:
        vars(VarList): list of problem variables (:class:`~mip.model.Var`)
        constrs(ConstrList): list of constraints (:class:`~mip.model.Constr`)

    Examples:
        >>> from mip import Model, MAXIMIZE, CBC, INTEGER, OptimizationStatus
        >>> model = Model(sense=MAXIMIZE, solver_name=CBC)
        >>> x = model.add_var(name='x', var_type=INTEGER, lb=0, ub=10)
        >>> y = model.add_var(name='y', var_type=INTEGER, lb=0, ub=10)
        >>> model += x + y <= 10
        >>> model.objective = x + y
        >>> status = model.optimize(max_seconds=2)
        >>> status == OptimizationStatus.OPTIMAL
        True
    """

    def __init__(
        self: "Model",
        name: str = "",
        sense: str = MINIMIZE,
        solver_name: str = "",
        solver: Optional[Solver] = None,
    ):
        """Model constructor

        Creates a Mixed-Integer Linear Programming Model. The default model
        optimization direction is Minimization. To store and optimize the model
        the MIP package automatically searches and connects in runtime to the
        dynamic library of some MIP solver installed on your computer, nowadays
        gurobi and cbc are supported. This solver is automatically selected,
        but you can force the selection of a specific solver with the parameter
        solver_name.

        Args:
            name (str): model name
            sense (str): MINIMIZATION ("MIN") or MAXIMIZATION ("MAX")
            solver_name(str): gurobi or cbc, searches for which
                solver is available if not informed
            solver(Solver): a (:class:`~mip.solver.Solver`) object; note that
                if this argument is provided, solver_name will be ignored
        """
        self._ownSolver = True
        # initializing variables with default values
        self.solver_name = solver_name
        self.solver = solver  # type: Optional[Solver]

        # reading solver_name from an environment variable (if applicable)
        if not solver:
            if not self.solver_name and "solver_name" in environ:
                self.solver_name = environ["solver_name"]
            if not self.solver_name and "solver_name".upper() in environ:
                self.solver_name = environ["solver_name".upper()]

            # creating a solver instance
            if self.solver_name.upper() in ["GUROBI", "GRB"]:
                from mip.gurobi import SolverGurobi

                self.solver = SolverGurobi(self, name, sense)
            elif self.solver_name.upper() == "CBC":
                from mip.cbc import SolverCbc

                self.solver = SolverCbc(self, name, sense)
            else:
                # checking which solvers are available
                try:
                    from mip.gurobi import SolverGurobi

                    has_gurobi = True
                except ImportError:
                    has_gurobi = False

                if has_gurobi:
                    from mip.gurobi import SolverGurobi

                    self.solver = SolverGurobi(self, name, sense)
                    self.solver_name = GUROBI
                else:
                    from mip.cbc import SolverCbc

                    self.solver = SolverCbc(self, name, sense)
                    self.solver_name = CBC

        # list of constraints and variables
        self.constrs = ConstrList(self)
        self.vars = VarList(self)

        self._status = OptimizationStatus.LOADED

        # initializing additional control variables
        self.__cuts = -1
        self.__cut_passes = -1
        self.__clique = -1
        self.__preprocess = -1
        self.__cuts_generator = None
        self.__lazy_constrs_generator = None
        self.__start = None
        self.__threads = 0
        self.__lp_method = LP_Method.AUTO
        self.__n_cols = 0
        self.__n_rows = 0
        self.__gap = INF
        self.__store_search_progress_log = False
        self.__plog = ProgressLog()
        self.__integer_tol = 1e-6
        self.__infeas_tol = 1e-6
        self.__opt_tol = 1e-6
        self.__max_mip_gap = 1e-4
        self.__max_mip_gap_abs = 1e-10

    def __del__(self: "Model"):
        del self.solver

    def __iadd__(self: "Model", other) -> "Model":
        if isinstance(other, LinExpr):
            if len(other.sense) == 0:
                # adding objective function components
                self.objective = other
            else:
                # adding constraint
                self.add_constr(other)
        elif isinstance(other, tuple):
            if isinstance(other[0], LinExpr) and isinstance(other[1], str):
                if len(other[0].sense) == 0:
                    self.objective = other[0]
                else:
                    self.add_constr(other[0], other[1])
        elif isinstance(other, CutPool):
            for cut in other.cuts:
                self.add_constr(cut)

        return self

    def add_var(
        self: "Model",
        name: str = "",
        lb: float = 0.0,
        ub: float = INF,
        obj: float = 0.0,
        var_type: str = CONTINUOUS,
        column: Column = None,
    ) -> Var:
        """ Creates a new variable in the model, returning its reference

        Args:
            name (str): variable name (optional)
            lb (float): variable lower bound, default 0.0
            ub (float): variable upper bound, default infinity
            obj (float): coefficient of this variable in the objective
              function, default 0
            var_type (str): CONTINUOUS ("C"), BINARY ("B") or INTEGER ("I")
            column (Column): constraints where this variable will appear,
                necessary only when constraints are already created in
                the model and a new variable will be created.

        Examples:

            To add a variable :code:`x` which is continuous and greater or
            equal to zero to model :code:`m`::

                x = m.add_var()

            The following code creates a vector of binary variables
            :code:`x[0], ..., x[n-1]` to model :code:`m`::

                x = [m.add_var(var_type=BINARY) for i in range(n)]
        """
        return self.vars.add(name, lb, ub, obj, var_type, column)

    def add_constr(self: "Model", lin_expr: LinExpr, name: str = "") -> Constr:
        r"""Creates a new constraint (row).

        Adds a new constraint to the model, returning its reference.

        Args:
            lin_expr(LinExpr): linear expression
            name(str): optional constraint name, used when saving model to\
            lp or mps files

        Examples:

        The following code adds the constraint :math:`x_1 + x_2 \leq 1`
        (x1 and x2 should be created first using
        :func:`add_var<mip.model.Model.add_var>`)::

            m += x1 + x2 <= 1

        Which is equivalent to::

            m.add_constr( x1 + x2 <= 1 )

        Summation expressions can be used also, to add the constraint \
        :math:`\displaystyle \sum_{i=0}^{n-1} x_i = y` and name this \
        constraint :code:`cons1`::

            m += xsum(x[i] for i in range(n)) == y, "cons1"

        Which is equivalent to::

            m.add_constr( xsum(x[i] for i in range(n)) == y, "cons1" )
        """

        if isinstance(lin_expr, bool):
            raise InvalidLinExpr(
                "A boolean (true/false) cannot be " "used as a constraint."
            )
        return self.constrs.add(lin_expr, name)

    def add_lazy_constr(self: "Model", expr: LinExpr):
        """Adds a lazy constraint

           A lazy constraint is a constraint that is only inserted
           into the model after the first integer solution that violates
           it is found. When lazy constraints are used a restricted
           pre-processing is executed since the complete model is not
           available at the beginning. If the number of lazy constraints
           is too large then they can be added during the search process
           by implementing a
           :class:`~mip.callbacks.ConstrsGenerator` and setting the
           property :attr:`~mip.model.Model.lazy_constrs_generator` of
           :class:`~mip.model.Model`.

           Args:
               expr(LinExpr): the linear constraint
        """
        self.solver.add_lazy_constr(expr)

    def add_sos(self: "Model", sos: List[Tuple[Var, float]], sos_type: int):
        r"""Adds an Special Ordered Set (SOS) to the model

        In models with binary variables it is often the case that from a list
        of variables only one can receive value 1 in a feasible solution. When
        large constraints of this type exist (packing and partitioning),
        branching in one variable at time usually doesn't work well: while
        fixing one of these variables to one leaves only one possible feasible
        value for the other variables in this set (zero), fixing one variable
        to zero keeps all other variables free. This *unbalanced* branching is
        highly ineffective. A Special ordered set (SOS) is a set
        :math:`\mathcal{S}=\{s_1, s_2, \ldots, s_k\}` with weights
        :math:`[w_1, w_2, \ldots, w_k] \in \mathbb{R}^+`. With this structure
        available branching on a fractional solution :math:`x^*` for these
        variables can be performed computing:


        .. math::

            \min \{ u_{k'} : u_{k'} = | \sum_{j=1\,\ldots \,k'-1}
            w_j \ldotp x^*_j - \sum_{j=k'\,\ldots ,k} w_j \ldotp x^*_j | \}


        Then, branching :math:`\mathcal{S}_1` would be
        :math:`\displaystyle \sum_{j=1, \ldots, k'-1} x_j = 0`
        and
        :math:`\displaystyle \mathcal{S}_2 = \sum_{j=k', \ldots, k} x_j = 0`.

        Args:
            sos(List[Tuple[Var, float]]):
                list including variables (not necessarily binary) and
                respective weights in the model
            sos_type(int):
                1 for Type 1 SOS, where at most one of the binary
                variables can be set to one and 2 for Type 2 SOS, where at
                most two variables from the list may be selected. In type
                2 SOS the two selected variables will be consecutive in
                the list.
        """
        self.solver.add_sos(sos, sos_type)

    def clear(self: "Model"):
        """Clears the model

        All variables, constraints and parameters will be reset. In addition,
        a new solver instance will be instantiated to implement the
        formulation.
        """
        # creating a new solver instance
        sense = self.sense

        if self.solver_name in [GUROBI, "gurobi"]:
            from mip.gurobi import SolverGurobi

            self.solver = SolverGurobi(self, self.name, sense)
        elif self.solver_name.upper() == CBC:
            from mip.cbc import SolverCbc

            self.solver = SolverCbc(self, self.name, sense)
        else:
            # checking which solvers are available
            from mip import gurobi

            if gurobi.found:
                from mip.gurobi import SolverGurobi

                self.solver = SolverGurobi(self, self.name, sense)
                self.solver_name = GUROBI
            else:
                from mip.cbc import SolverCbc

                self.solver = SolverCbc(self, self.name, sense)
                self.solver_name = CBC

        # list of constraints and variables
        self.constrs = ConstrList(self)
        self.vars = VarList(self)

        # initializing additional control variables
        self.__cuts = 1
        self.__cuts_generator = None
        self.__lazy_constrs_generator = None
        self.__start = []
        self._status = OptimizationStatus.LOADED
        self.__threads = 0

    def copy(self: "Model", solver_name: str = "") -> "Model":
        """ Creates a copy of the current model

        Args:
            solver_name(str): solver name (optional)

        Returns:
            clone of current model
        """
        if not solver_name:
            solver_name = self.solver_name
        copy = Model(self.name, self.sense, solver_name)

        # adding variables
        for v in self.vars:
            copy.add_var(
                name=v.name, lb=v.lb, ub=v.ub, obj=v.obj, var_type=v.var_type
            )

        # adding constraints
        for c in self.constrs:
            orig_expr = c.expr
            expr = LinExpr(const=orig_expr.const, sense=orig_expr.sense)
            for (var, value) in orig_expr.expr.items():
                expr.add_term(self.vars[var.idx], value)
            copy.add_constr(lin_expr=expr, name=c.name)

        # setting objective function"s constant
        copy.objective_const = self.objective_const

        return copy

    def constr_by_name(self: "Model", name: str) -> Optional[Constr]:
        """ Queries a constraint by its name

        Args:
            name(str): constraint name

        Returns:
            constraint or None if not found
        """
        cidx = self.solver.constr_get_index(name)
        if cidx < 0 or cidx > len(self.constrs):
            return None
        return self.constrs[cidx]

    def var_by_name(self: "Model", name: str) -> Optional[Var]:
        """Searchers a variable by its name

        Returns:
            Variable or None if not found
        """
        v = self.solver.var_get_index(name)
        if v < 0 or v > len(self.vars):
            return None
        return self.vars[v]

    def generate_cuts(
        self: "Model",
        cut_types: Optional[List[CutType]] = None,
        max_cuts: int = 8192,
        min_viol: float = 1e-4,
    ) -> CutPool:
        """Tries to generate cutting planes for the current fractional
        solution. To optimize only the linear programming relaxation and not
        discard integrality information from variables you must call first
        :code:`model.optimize(relax=True)`.

        This method only works with the CBC mip solver, as Gurobi does not
        supports calling only cut generators.

        Args:
            cut_types (List[CutType]): types of cuts that can be generated, if
                an empty list is specified then all available cut generators
                will be called.
            max_cuts(int): cut separation will stop when at least max_cuts
                violated cuts were found.
            min_viol(float): cuts which are not violated by at least min_viol
                will be discarded.


        """
        if self.status != OptimizationStatus.OPTIMAL:
            raise SolutionNotAvailable()

        return self.solver.generate_cuts(cut_types, max_cuts)

    def optimize(
        self: "Model",
        max_seconds: float = INF,
        max_nodes: int = INF,
        max_solutions: int = INF,
        relax: bool = False,
    ) -> OptimizationStatus:
        """ Optimizes current model

        Optimizes current model, optionally specifying processing limits.

        To optimize model :code:`m` within a processing time limit of
        300 seconds::

            m.optimize(max_seconds=300)

        Args:
            max_seconds (float): Maximum runtime in seconds (default: inf)
            max_nodes (float): Maximum number of nodes (default: inf)
            max_solutions (float): Maximum number of solutions (default: inf)
            relax (bool): if true only the linear programming relaxation will
                be solved, i.e. integrality constraints will be temporarily
                discarded.

        Returns:
            optimization status, which can be OPTIMAL(0), ERROR(-1),
            INFEASIBLE(1), UNBOUNDED(2). When optimizing problems
            with integer variables some additional cases may happen,
            FEASIBLE(3) for the case when a feasible solution was found
            but optimality was not proved, INT_INFEASIBLE(4) for the case
            when the lp relaxation is feasible but no feasible integer
            solution exists and NO_SOLUTION_FOUND(5) for the case when
            an integer solution was not found in the optimization.

        """
        if not self.solver.num_cols():
            print("Model has no variables. Nothing to optimize.")
            return OptimizationStatus.OTHER

        if self.__threads != 0:
            self.solver.set_num_threads(self.__threads)
        # self.solver.set_callbacks(branch_selector,
        # incumbent_updater, lazy_constrs_generator)
        self.solver.set_processing_limits(
            max_seconds, max_nodes, max_solutions
        )

        self._status = self.solver.optimize(relax)
        # has a solution and is a MIP
        if self.num_solutions and self.num_int > 0:
            best = self.objective_value
            lb = self.objective_bound
            if abs(best) <= 1e-10:
                self.__gap = INF
            else:
                self.__gap = abs(best - lb) / abs(best)

        if self.store_search_progress_log:
            self.__plog.log = self.solver.get_log()
            self.__plog.instance = self.name

        return self._status

    def read(self: "Model", path: str):
        """Reads a MIP model or an initial feasible solution.

           One of  the following file name extensions should be used
           to define the contents of what will be loaded:

           :code:`.lp`
             mip model stored in the
             `LP file format <https://www.ibm.com/support/knowledgecenter/SSSA5P_12.9.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/tutorials/InteractiveOptimizer/usingLPformat.html>`_

           :code:`.mps`
             mip model stored in the
             `MPS file format <https://en.wikipedia.org/wiki/MPS_(format)>`_

           :code:`.sol`
             initial feasible solution

        Note: if a new problem is readed, all variables, constraints
        and parameters from the current model will be cleared.

        Args:
            path(str): file name
        """
        if not isfile(path):
            raise OSError(2, "File {} does not exists".format(path))

        if path.lower().endswith(".sol") or path.lower().endswith(".mst"):
            mip_start = load_mipstart(path)
            if not mip_start:
                raise Exception(
                    "File {} does not contains a valid feasible \
                                 solution.".format(
                        path
                    )
                )
            var_list = []
            for name, value in mip_start:
                var = self.var_by_name(name)
                if var is not None:
                    var_list.append((var, value))
            if not var_list:
                raise Exception(
                    "Invalid variable(s) name(s) in \
                                 mipstart file {}".format(
                        path
                    )
                )

            self.start = var_list
            return

        # reading model
        model_ext = [".lp", ".mps", ".mps.gz"]

        fn_low = path.lower()
        for ext in model_ext:
            if fn_low.endswith(ext):
                self.clear()
                self.solver.read(path)
                self.vars.update_vars(self.solver.num_cols())
                self.constrs.update_constrs(self.solver.num_rows())
                return

        raise Exception(
            "Use .lp, .mps, .sol or .mst as file extension \
                         to indicate the file format."
        )

    def relax(self: "Model"):
        """ Relax integrality constraints of variables

        Changes the type of all integer and binary variables to
        continuous. Bounds are preserved.
        """
        self.solver.relax()

    def write(self: "Model", file_path: str):
        """Saves a MIP model or an initial feasible solution.

           One of  the following file name extensions should be used
           to define the contents of what will be saved:

           :code:`.lp`
             mip model stored in the
             `LP file format <https://www.ibm.com/support/knowledgecenter/SSSA5P_12.9.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/tutorials/InteractiveOptimizer/usingLPformat.html>`_

           :code:`.mps`
             mip model stored in the
             `MPS file format <https://en.wikipedia.org/wiki/MPS_(format)>`_

           :code:`.sol`
             initial feasible solution

        Args:
            file_path(str): file name
        """
        if file_path.lower().endswith(".sol") or file_path.lower().endswith(
            ".mst"
        ):
            if self.start:
                save_mipstart(self.start, file_path)
            else:
                mip_start = [
                    (var, var.x) for var in self.vars if abs(var.x) >= 1e-8
                ]
                save_mipstart(mip_start, file_path)
        elif file_path.lower().endswith(".lp") or file_path.lower().endswith(
            ".mps"
        ):
            self.solver.write(file_path)
        else:
            raise Exception(
                "Use .lp, .mps, .sol or .mst as file extension \
                             to indicate the file format."
            )

    @property
    def objective_bound(self: "Model") -> Optional[float]:
        """
            A valid estimate computed for the optimal solution cost,
            lower bound in the case of minimization, equals to
            :attr:`~mip.model.Model.objective_value` if the
            optimal solution was found.
        """
        if self.status not in [
            OptimizationStatus.OPTIMAL,
            OptimizationStatus.FEASIBLE,
            OptimizationStatus.NO_SOLUTION_FOUND,
        ]:
            return None

        return self.solver.get_objective_bound()

    @property
    def name(self: "Model") -> str:
        """The problem (instance) name

           This name should be used to identify the instance that this model
           refers, e.g.: productionPlanningMay19. This name is stored when
           saving (:meth:`~mip.model.Model.write`) the model in :code:`.LP`
           or :code:`.MPS` file formats.
        """
        return self.solver.get_problem_name()

    @name.setter
    def name(self: "Model", name: str):
        self.solver.set_problem_name(name)

    @property
    def objective(self: "Model") -> LinExpr:
        """The objective function of the problem as a linear expression.

        Examples:

            The following code adds all :code:`x` variables :code:`x[0],
            ..., x[n-1]`, to the objective function of model :code:`m`
            with the same cost :code:`w`::

                m.objective = xsum(w*x[i] for i in range(n))

            A simpler way to define the objective function is the use of the
            model operator += ::

                m += xsum(w*x[i] for i in range(n))

            Note that the only difference of adding a constraint is the lack of
            a sense and a rhs.
        """
        return self.solver.get_objective()

    @objective.setter
    def objective(self: "Model", objective: Union[int, float, Var, LinExpr]):
        if isinstance(objective, (int, float)):
            self.solver.set_objective(LinExpr([], [], objective))
        elif isinstance(objective, Var):
            self.solver.set_objective(LinExpr([objective], [1]))
        elif isinstance(objective, LinExpr):
            self.solver.set_objective(objective)

    @property
    def verbose(self: "Model") -> int:
        """0 to disable solver messages printed on the screen, 1 to enable
        """
        return self.solver.get_verbose()

    @verbose.setter
    def verbose(self: "Model", verbose: int):
        self.solver.set_verbose(verbose)

    @property
    def lp_method(self: "Model") -> LP_Method:
        """Which  method should be used to solve the linear programming
        problem. If the problem has integer variables that this affects only
        the solution of the first linear programming relaxation."""
        return self.__lp_method

    @lp_method.setter
    def lp_method(self: "Model", lpm: LP_Method):
        self.__lp_method = lpm

    @property
    def threads(self: "Model") -> int:
        r"""number of threads to be used when solving the problem.
        0 uses solver default configuration, -1 uses the number of available
        processing cores and :math:`\geq 1` uses the specified number of
        threads. An increased number of threads may improve the solution
        time but also increases the memory consumption."""
        return self.__threads

    @threads.setter
    def threads(self: "Model", threads: int):
        self.__threads = threads

    @property
    def sense(self: "Model") -> str:
        """ The optimization sense

        Returns:
            the objective function sense, MINIMIZE (default) or (MAXIMIZE)
        """

        return self.solver.get_objective_sense()

    @sense.setter
    def sense(self: "Model", sense: str):
        self.solver.set_objective_sense(sense)

    @property
    def objective_const(self: "Model") -> float:
        """Returns the constant part of the objective function
        """
        return self.solver.get_objective_const()

    @objective_const.setter
    def objective_const(self: "Model", objective_const: float):
        self.solver.set_objective_const(objective_const)

    @property
    def objective_value(self: "Model") -> Optional[float]:
        """Objective function value of the solution found or None
        if model was not optimized
        """
        if self.status not in [
            OptimizationStatus.OPTIMAL,
            OptimizationStatus.FEASIBLE,
        ]:
            return None
        return self.solver.get_objective_value()

    @property
    def gap(self: "Model") -> float:
        r"""
           The optimality gap considering the cost of the best solution found
           (:attr:`~mip.model.Model.objective_value`)
           :math:`b` and the best objective bound :math:`l`
           (:attr:`~mip.model.Model.objective_bound`) :math:`g` is
           computed as: :math:`g=\\frac{|b-l|}{|b|}`.
           If no solution was found or if :math:`b=0` then :math:`g=\infty`.
           If the optimal solution was found then :math:`g=0`.
        """
        return self.__gap

    @property
    def search_progress_log(self: "Model") -> ProgressLog:
        """
            Log of bound improvements in the search.
            The output of MIP solvers is a sequence of improving
            incumbent solutions (primal bound) and estimates for the optimal
            cost (dual bound). When the costs of these two bounds match the
            search is concluded. In truncated searches, the most common
            situation for hard problems, at the end of the search there is a
            :attr:`~mip.model.Model.gap` between these bounds. This
            property stores the detailed events of improving these
            bounds during the search process. Analyzing the evolution
            of these bounds you can see if you need to improve your
            solver w.r.t. the production of feasible solutions, by including an
            heuristic to produce a better initial feasible solution, for
            example, or improve the formulation with cutting planes, for
            example, to produce better dual bounds. To enable storing the
            :attr:`~mip.model.Model.search_progress_log` set
            :attr:`~mip.model.Model.store_search_progress_log` to True.
        """
        return self.__plog

    @property
    def store_search_progress_log(self: "Model") -> bool:
        """
            Wether :attr:`~mip.model.Model.search_progress_log` will be stored
            or not when optimizing. Default False. Activate it if you want to
            analyze bound improvements over time."""
        return self.__store_search_progress_log

    @store_search_progress_log.setter
    def store_search_progress_log(self: "Model", store: bool):
        self.__store_search_progress_log = store

    # def plot_bounds_evolution(self):
    #    import matplotlib.pyplot as plt
    #    log = self.search_progress_log
    #
    #    # plotting lower bound
    #    x = [a[0] for a in log]
    #    y = [a[1][0] for a in log]
    #    plt.plot(x, y)
    #    # plotting upper bound
    #    x = [a[0] for a in log if a[1][1] < 1e+50]
    #    y = [a[1][1] for a in log if a[1][1] < 1e+50]
    #    plt.plot(x, y)
    #    plt.show()

    @property
    def num_solutions(self: "Model") -> int:
        """Number of solutions found during the MIP search

        Returns:
            number of solutions stored in the solution pool

        """
        return self.solver.get_num_solutions()

    @property
    def objective_values(self: "Model") -> List[float]:
        """List of costs of all solutions in the solution pool

        Returns:
            costs of all solutions stored in the solution pool
            as an array from 0 (the best solution) to
            :attr:`~mip.model.Model.num_solutions`-1.
        """
        return [
            float(self.solver.get_objective_value_i(i))
            for i in range(self.num_solutions)
        ]

    @property
    def cuts_generator(self: "Model") -> "ConstrsGenerator":
        """A cuts generator is an :class:`~mip.callbacks.ConstrsGenerator`
        object that receives a fractional solution and tries to generate one or
        more constraints (cuts) to remove it. The cuts generator is called in
        every node of the branch-and-cut tree where a solution that violates
        the integrality constraint of one or more variables is found.
        """

        return self.__cuts_generator

    @cuts_generator.setter
    def cuts_generator(self: "Model", cuts_generator: ConstrsGenerator):
        self.__cuts_generator = cuts_generator

    @property
    def lazy_constrs_generator(self: "Model") -> "ConstrsGenerator":
        """A lazy constraints generator is an
        :class:`~mip.callbacks.ConstrsGenerator` object that receives
        an integer solution and checks its feasibility. If
        the solution is not feasible then one or more constraints can be
        generated to remove it. When a lazy constraints generator is informed
        it is assumed that the initial formulation is incomplete. Thus, a
        restricted pre-processing routine may be applied. If the initial
        formulation is incomplete, it may be interesting to use the same
        :class:`~mip.callbacks.ConstrsGenerator` to generate cuts *and* lazy
        constraints. The use of *only* lazy constraints may be useful then
        integer solutions rarely violate these constraints.
        """

        return self.__lazy_constrs_generator

    @lazy_constrs_generator.setter
    def lazy_constrs_generator(
        self: "Model", lazy_constrs_generator: ConstrsGenerator
    ):
        self.__lazy_constrs_generator = lazy_constrs_generator

    @property
    def emphasis(self: "Model") -> SearchEmphasis:
        """defines the main objective of the search, if set to 1 (FEASIBILITY)
        then the search process will focus on try to find quickly feasible
        solutions and improving them; if set to 2 (OPTIMALITY) then the
        search process will try to find a provable optimal solution,
        procedures to further improve the lower bounds will be activated in
        this setting, this may increase the time to produce the first
        feasible solutions but will probably pay off in longer runs;
        the default option if 0, where a balance between optimality and
        feasibility is sought.
        """
        return self.solver.get_emphasis()

    @emphasis.setter
    def emphasis(self: "Model", emphasis: SearchEmphasis):
        self.solver.set_emphasis(emphasis)

    @property
    def preprocess(self: "Model") -> int:
        """Enables/disables pre-processing. Pre-processing tries to improve
        your MIP formulation. -1 means automatic, 0 means off and 1
        means on."""
        return self.__preprocess

    @preprocess.setter
    def preprocess(self: "Model", prep: int):
        self.__preprocess = prep

    @property
    def pump_passes(self: "Model") -> int:
        """Number of passes of the Feasibility Pump [FGL05]_ heuristic.
           You may increase this value if you are not getting feasible
           solutions."""
        return self.solver.get_pump_passes()

    @pump_passes.setter
    def pump_passes(self: "Model", passes: int):
        self.solver.set_pump_passes(passes)

    @property
    def cuts(self: "Model") -> int:
        """Controls the generation of cutting planes, -1 means automatic, 0
        disables completely, 1 (default) generates cutting planes in a moderate
        way, 2 generates cutting planes aggressively and 3 generates even more
        cutting planes. Cutting planes usually improve the LP relaxation bound
        but also make the solution time of the LP relaxation larger, so the
        overall effect is hard to predict and experimenting different values
        for this parameter may be beneficial."""

        return self.__cuts

    @cuts.setter
    def cuts(self: "Model", gencuts: int):
        self.__cuts = gencuts

    @property
    def cut_passes(self: "Model") -> int:
        """Maximum number of rounds of cutting planes. You may set this
        parameter to low values if you see that a significant amount of
        time is being spent generating cuts without any improvement in
        the lower bound. -1 means automatic, values greater than zero
        specify the maximum number of rounds."""
        return self.__cut_passes

    @cut_passes.setter
    def cut_passes(self: "Model", cp: int):
        self.__cut_passes = cp

    @property
    def clique(self: "Model") -> int:
        """Controls the generation of clique cuts. -1 means automatic,
        0 disables it, 1 enables it and 2 enables more aggressive clique
        generation."""
        return self.__clique

    @clique.setter
    def clique(self: "Model", clq: int):
        self.__clique = clq

    @property
    def start(self: "Model") -> List[Tuple[Var, float]]:
        """Initial feasible solution

        Enters an initial feasible solution. Only the main binary/integer
        decision variables which appear with non-zero values in the initial
        feasible solution need to be informed. Auxiliary or continuous
        variables are automatically computed.
        """
        return self.__start

    @start.setter
    def start(self: "Model", start: List[Tuple[Var, float]]):
        self.__start = start
        self.solver.set_start(start)

    def validate_mip_start(self: "Model"):
        """Validates solution entered in MIPStart

        If the solver engine printed messages indicating that the initial
        feasible solution that you entered in :attr:`~mip.model.start` is not
        valid then you can call this method to help discovering which set of
        variables is causing infeasibility. The current version is quite
        simple: the model is relaxed and one variable entered in mipstart is
        fixed per iteration, indicating if the model still feasible or not.
        """

        out.write("Checking feasibility of MIPStart\n")
        mc = self.copy()
        mc.verbose = 0
        mc.relax()
        mc.optimize()
        if mc.status == OptimizationStatus.INFEASIBLE:
            out.write("Model is infeasible.\n")
            return
        if mc.status == OptimizationStatus.UNBOUNDED:
            out.write(
                "Model is unbounded. You probably need to insert "
                "additional constraints or bounds in variables.\n"
            )
            return
        if mc.status != OptimizationStatus.OPTIMAL:
            print(
                "Unexpected status while optimizing LP relaxation:"
                " {}".format(mc.status)
            )

        print("Model LP relaxation bound is {}".format(mc.objective_value))

        for (var, value) in self.start:
            out.write("\tfixing %s to %g ... " % (var.name, value))
            mc += var == value
            mc.optimize()
            if mc.status == OptimizationStatus.OPTIMAL:
                print("ok, obj now: {}".format(mc.objective_value))
            else:
                print("NOT OK, optimization status: {}".format(mc.status))
                return

        print(
            "Linear Programming relaxation of model with fixations from "
            "MIPStart is feasible."
        )
        print("MIP model may still be infeasible.")

    @property
    def num_cols(self: "Model") -> int:
        """number of columns (variables) in the model"""
        return len(self.vars)

    @property
    def num_int(self: "Model") -> int:
        """number of integer variables in the model"""
        return self.solver.num_int()

    @property
    def num_rows(self: "Model") -> int:
        """number of rows (constraints) in the model"""
        return len(self.constrs)

    @property
    def num_nz(self: "Model") -> int:
        """number of non-zeros in the constraint matrix"""
        return self.solver.num_nz()

    @property
    def cutoff(self: "Model") -> float:
        """upper limit for the solution cost, solutions with cost > cutoff
        will be removed from the search space, a small cutoff value may
        significantly speedup the search, but if cutoff is set to a value too
        low the model will become infeasible"""
        return self.solver.get_cutoff()

    @cutoff.setter
    def cutoff(self: "Model", cutoff: float):
        self.solver.set_cutoff(cutoff)

    @property
    def integer_tol(self: "Model") -> float:
        """Maximum distance to the nearest integer for a variable to be
        considered with an integer value. Default value: 1e-6. Tightening this
        value can increase the numerical precision but also probably increase
        the running time. As floating point computations always involve some
        loss of precision, values too close to zero will likely render some
        models impossible to optimize."""
        return self.__integer_tol

    @integer_tol.setter
    def integer_tol(self: "Model", int_tol: float):
        self.__integer_tol = int_tol

    @property
    def infeas_tol(self: "Model") -> float:
        """Maximum allowed violation for constraints. Default value: 1e-6.
        Tightening this value can increase the numerical precision but also
        probably increase the running time. As floating point computations
        always involve some loss of precision, values too close to zero will
        likely render some models impossible to optimize."""

        return self.__infeas_tol

    @infeas_tol.setter
    def infeas_tol(self: "Model", inf_tol: float):
        self.__infeas_tol = inf_tol

    @property
    def opt_tol(self: "Model") -> float:
        """Maximum reduced cost value for a solution of the LP relaxation to be
        considered optimal. Default value: 1e-6.  Tightening this value can
        increase the numerical precision but also probably increase the running
        time. As floating point computations always involve some loss of
        precision, values too close to zero will likely render some models
        impossible to optimize."""
        return self.__opt_tol

    @opt_tol.setter
    def opt_tol(self: "Model", tol: float):
        self.__opt_tol = tol

    @property
    def max_mip_gap_abs(self: "Model") -> float:
        """Tolerance for the quality of the optimal solution, if a solution
        with cost :math:`c` and a lower bound :math:`l` are available and
        :math:`c-l<` :code:`mip_gap_abs`, the search will be concluded, see
        :attr:`~mip.model.Model.max_mip_gap` to determine a percentage value.
        Default value: 1e-10."""
        return self.__max_mip_gap_abs

    @max_mip_gap_abs.setter
    def max_mip_gap_abs(self: "Model", max_mip_gap_abs: float):
        self.__max_mip_gap_abs = max_mip_gap_abs

    @property
    def max_mip_gap(self: "Model") -> float:
        """value indicating the tolerance for the maximum percentage deviation
        from the optimal solution cost, if a solution with cost :math:`c` and
        a lower bound :math:`l` are available and
        :math:`(c-l)/l <` :code:`max_mip_gap` the search will be concluded.
        Default value: 1e-4."""
        return self.__max_mip_gap

    @max_mip_gap.setter
    def max_mip_gap(self: "Model", max_mip_gap: float):
        self.__max_mip_gap = max_mip_gap

    @property
    def max_seconds(self: "Model") -> float:
        """time limit in seconds for search"""
        return self.solver.get_max_seconds()

    @max_seconds.setter
    def max_seconds(self: "Model", max_seconds: float):
        self.solver.set_max_seconds(max_seconds)

    @property
    def max_nodes(self: "Model") -> int:
        """maximum number of nodes to be explored in the search tree"""
        return self.solver.get_max_nodes()

    @max_nodes.setter
    def max_nodes(self: "Model", max_nodes: int):
        self.solver.set_max_nodes(max_nodes)

    @property
    def max_solutions(self: "Model") -> int:
        """solution limit, search will be stopped when :code:`max_solutions`
        were found"""
        return self.solver.get_max_solutions()

    @max_solutions.setter
    def max_solutions(self: "Model", max_solutions: int):
        self.solver.set_max_solutions(max_solutions)

    @property
    def status(self: "Model") -> OptimizationStatus:
        """ optimization status, which can be OPTIMAL(0), ERROR(-1),
        INFEASIBLE(1), UNBOUNDED(2). When optimizing problems
        with integer variables some additional cases may happen, FEASIBLE(3)
        for the case when a feasible solution was found but optimality was
        not proved, INT_INFEASIBLE(4) for the case when the lp relaxation is
        feasible but no feasible integer solution exists and
        NO_SOLUTION_FOUND(5) for the case when an integer solution was not
        found in the optimization.
        """
        return self._status

    def add_cut(self: "Model", cut: LinExpr):

        """Adds a violated inequality (cutting plane) to the linear programming
        model. If called outside the cut callback performs exactly as
        :meth:`~mip.model.Model.add_constr`. When called inside the cut
        callback the cut is included in the solver's cut pool, which will later
        decide if this cut should be added or not to the model. Repeated cuts,
        or cuts which will probably be less effective, e.g. with a very small
        violation, can be discarded.

        Args:
            cut(LinExpr): violated inequality
        """
        self.solver.add_cut(cut)

    def remove(self: "Model", objects):
        """removes variable(s) and/or constraint(s) from the model

        Args:
            objects: can be a Var, a Constr or a list of these objects
        """
        if isinstance(objects, Var) or isinstance(objects, Constr):
            objects = [objects]

        if isinstance(objects, list):
            vlist = []
            clist = []
            for o in objects:
                if isinstance(o, Var):
                    vlist.append(o)
                elif isinstance(o, Constr):
                    clist.append(o)
                else:
                    raise Exception(
                        "Cannot handle removal of object of type "
                        + type(o)
                        + " from model."
                    )
            if vlist:
                self.vars.remove(vlist)
            if clist:
                self.constrs.remove(clist)
        else:
            raise Exception(
                "Cannot handle removal of object of type "
                + type(objects)
                + " from model."
            )

    def translate(self: "Model", ref) -> Union[List[Any], Dict[Any, Any], Var]:
        """Translates references of variables/containers of variables
        from another model to this model. Can be used to translate
        references of variables in the original model to references
        of variables in the pre-processed model."""

        res = None  # type: Union[List[Any], Dict[Any, Any], Var]

        if isinstance(ref, Var):
            return self.var_by_name(ref.name)
        if isinstance(ref, list):
            res = list()
            for el in ref:
                res.append(self.translate(el))
            return res
        if isinstance(ref, dict):
            res = dict()
            for key, value in ref.items():
                res[key] = self.translate(value)
            return res

        return ref

    def check_optimization_results(self):
        """Checks the consistency of the optimization results, i.e., if the
        solution(s) produced by the MIP solver respect all constraints and
        variable values are within acceptable bounds and are integral when
        requested"""
        if self.status in [
            OptimizationStatus.FEASIBLE,
            OptimizationStatus.OPTIMAL,
        ]:
            assert self.num_solutions >= 1
        if self.num_solutions or self.status in [
            OptimizationStatus.FEASIBLE,
            OptimizationStatus.OPTIMAL,
        ]:
            if self.sense == MINIMIZE:
                assert self.objective_bound <= self.objective_value + 1e-10
            else:
                assert self.objective_bound + 1e-10 >= self.objective_value

            for c in self.constrs:
                if c.expr.violation >= self.infeas_tol + self.infeas_tol * 0.1:
                    raise InfeasibleSolution(
                        "Constraint {}:\n{}\n is violated."
                        "Computed violation is {}."
                        "Tolerance for infeasibility is {}."
                        "Solution status is {}.".format(
                            c.name,
                            str(c),
                            c.expr.violation,
                            self.infeas_tol,
                            self.status,
                        )
                    )
            for v in self.vars:
                if v.x <= v.lb - 1e-10 or v.x >= v.ub + 1e-10:
                    raise InfeasibleSolution(
                        "Invalid solution value for "
                        "variable {}={} variable bounds"
                        " are [{}, {}].".format(v.name, v.x, v.lb, v.ub)
                    )
                if v.var_type in [BINARY, INTEGER]:
                    if (
                        round(v.x) - v.x
                    ) >= self.integer_tol + self.integer_tol * 0.1:
                        raise InfeasibleSolution(
                            "Variable {}={} should be integral.".format(
                                v.name, v.x
                            )
                        )


def maximize(objective: Union[LinExpr, Var]) -> LinExpr:
    """
    Function that should be used to set the objective function to MAXIMIZE
    a given linear expression (passed as argument).

    Args:
        objective(Union[LinExpr, Var]): linear expression
    """
    if isinstance(objective, Var):
        objective = LinExpr([objective], [1.0])
    objective.sense = MAXIMIZE
    return objective


def minimize(objective: Union[LinExpr, Var]) -> LinExpr:
    """
    Function that should be used to set the objective function to MINIMIZE
    a given linear expression (passed as argument).

    Args:
        objective(Union[LinExpr, Var]): linear expression
    """
    if isinstance(objective, Var):
        objective = LinExpr([objective], [1.0])
    objective.sense = MINIMIZE
    return objective


def xsum(terms) -> LinExpr:
    """
    Function that should be used to create a linear expression from a
    summation. While the python function sum() can also be used, this
    function is optimized version for quickly generating the linear
    expression.

    Args:
        terms: set (ideally a list) of terms to be summed
    """
    result = LinExpr()
    for term in terms:
        result.add_term(term)
    return result


# function aliases
quicksum = xsum


def save_mipstart(sol: List[Tuple[Var, float]], file_name: str, obj=0.0):
    f = open(file_name, "w")
    f.write("Feasible solution - objective {}\n".format(obj))
    for i, (var, val) in enumerate(sol):
        f.write("{} {} {} {}\n".format(i, var.name, val, var.obj))
    f.close()


def load_mipstart(file_name: str) -> List[Tuple[str, float]]:
    f = open(file_name, "w")
    result = []
    line = f.next()
    for line in f:
        line = line.rstrip().lstrip().lower()
        line = " ".join(line.split())
        lc = line.split(" ")
        result.append((lc[1], float(lc[2])))
    return result


def read_custom_settings():
    global customCbcLib
    from pathlib import Path

    home = str(Path.home())
    import os

    config_path = os.path.join(home, ".config")
    if os.path.isdir(config_path):
        config_file = os.path.join(config_path, "python-mip")
        if os.path.isfile(config_file):
            f = open(config_file, "r")
            for line in f:
                if "=" in line:
                    cols = line.split("=")
                    if cols[0].strip().lower() == "cbc-library":
                        customCbcLib = (
                            cols[1].lstrip().rstrip().replace('"', "")
                        )


print("Using Python-MIP package version {}".format(VERSION))
customCbcLib = ""
read_custom_settings()

# vim: ts=4 sw=4 et
