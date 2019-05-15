from mip.callbacks import *
from mip.constants import *
from mip.exceptions import *
from math import inf
from typing import List, Tuple
from builtins import property
from os import environ
from collections.abc import Sequence


class Column:
    """A column contains all the non-zero entries of a variable in the
    constraint matrix. To create a variable see
    :meth:`~mip.model.model.add_var` """

    def __init__(self,
                 constrs: List["Constr"] = None,
                 coeffs: List[float] = None):
        self.constrs = constrs
        self.coeffs = coeffs


class Constr:
    """ A row (constraint) in the constraint matrix

        A constraint is a specific :class:`~mip.model.LinExpr`. Constraints
        can be added to the model using the overloaded operator
        +=, e.g., if :code:`m` is a model:

        .. code:: python

          m += 3*x1 + 4*x2 <= 5

        summation expressions are also supported:

        .. code:: python

          m += xsum(x[i] for i in range(n)) == 1
    """

    def __init__(self, model: "Model", idx: int):
        self.__model = model
        self.idx = idx

    def __hash__(self) -> int:
        return self.idx

    def __str__(self) -> str:
        if self.name:
            res = self.name+':'
        else:
            res = 'constr({}): '.format(self.idx+1)
        line = ''
        len_line = 0
        for (var, val) in self.expr.expr.items():
            astr = ' {:+} {}'.format(val, var.name)
            len_line += len(astr)
            line += astr

            if len_line > 75:
                line += '\n\t'
                len_line = 0
        res += line
        rhs = self.expr.const*-1.0
        if self.expr.sense == '=':
            res += ' = {}'.format(rhs)
        elif self.expr.sense == '<':
            res += ' <= {}'.format(rhs)
        elif self.expr.sense == '>':
            res += ' <= {}'.format(rhs)

        return res

    @property
    def pi(self) -> float:
        """value for the dual variable of this constraint in the optimal
        solution of a linear programming __model, cannot be evaluated for
        problems with integer variables"""
        return self.__model.solver.constr_get_pi(self)

    @property
    def expr(self) -> "LinExpr":
        """contents of the constraint"""
        return self.__model.solver.constr_get_expr(self)

    @expr.setter
    def expr(self, value: "LinExpr"):
        self.__model.solver.constr_set_expr(self, value)

    @property
    def name(self) -> str:
        """constraint name"""
        return self.__model.solver.constr_get_name(self.idx)


class LinExpr:
    """
    Linear expressions are used to enter the objective function and the model \
    constraints. These expressions are created using operators and variables.

    Consider a model object m, the objective function of :code:`m` can be
    specified as:

    .. code:: python

     m.objective = 10*x1 + 7*x4

    In the example bellow, a constraint is added to the model

    .. code:: python

     m += xsum(3*x[i] i in range(n)) - xsum(x[i] i in range(m))

    A constraint is just a linear expression with the addition of a sense (==,
    <= or >=) and a right hand side, e.g.:

    .. code:: python

     m += x1 + x2 + x3 == 1
    """

    def __init__(self,
                 variables: List["Var"] = None,
                 coeffs: List[float] = None,
                 const: float = 0.0,
                 sense: str = ""):
        self.__const = const
        self.__expr = {}
        self.__sense = sense

        if variables:
            assert len(variables) == len(coeffs)
            for i in range(len(coeffs)):
                if abs(coeffs[i]) <= 1e-12:
                    continue
                self.add_var(variables[i], coeffs[i])

    def __add__(self, other) -> "LinExpr":
        result = self.copy()
        if isinstance(other, Var):
            result.add_var(other, 1)
        elif isinstance(other, LinExpr):
            result.add_expr(other)
        elif isinstance(other, (int, float)):
            result.add_const(other)
        return result

    def __radd__(self, other) -> "LinExpr":
        return self.__add__(other)

    def __iadd__(self, other) -> "LinExpr":
        if isinstance(other, Var):
            self.add_var(other, 1)
        elif isinstance(other, LinExpr):
            self.add_expr(other)
        elif isinstance(other, (int, float)):
            self.add_const(other)
        return self

    def __sub__(self, other) -> "LinExpr":
        result = self.copy()
        if isinstance(other, Var):
            result.add_var(other, -1)
        elif isinstance(other, LinExpr):
            result.add_expr(other, -1)
        elif isinstance(other, (int, float)):
            result.add_const(-other)
        return result

    def __rsub__(self, other) -> "LinExpr":
        return (-self).__add__(other)

    def __isub__(self, other) -> "LinExpr":
        if isinstance(other, Var):
            self.add_var(other, -1)
        elif isinstance(other, LinExpr):
            self.add_expr(other, -1)
        elif isinstance(other, (int, float)):
            self.add_const(-other)
        return self

    def __mul__(self, other) -> "LinExpr":
        assert isinstance(other, (float, int))
        result = self.copy()
        result.__const *= other
        for var in result.__expr.keys():
            result.__expr[var] *= other

        # if constraint __sense will change
        if self.__sense == GREATER_OR_EQUAL and other <= -1e-8:
            self.__sense = LESS_OR_EQUAL
        if self.__sense == LESS_OR_EQUAL and other <= -1e-8:
            self.__sense = GREATER_OR_EQUAL

        return result

    def __rmul__(self, other) -> "LinExpr":
        return self.__mul__(other)

    def __imul__(self, other) -> "LinExpr":
        assert isinstance(other, (int, float))
        self.__const *= other
        for var in self.__expr.keys():
            self.__expr[var] *= other
        return self

    def __truediv__(self, other) -> "LinExpr":
        assert isinstance(other, (int, float))
        result = self.copy()
        result.__const /= other
        for var in result.__expr.keys():
            result.__expr[var] /= other
        return result

    def __itruediv__(self, other) -> "LinExpr":
        assert isinstance(other, int) or isinstance(other, float)
        self.__const /= other
        for var in self.__expr.keys():
            self.__expr[var] /= other
        return self

    def __neg__(self) -> "LinExpr":
        return self.__mul__(-1)

    def __str__(self) -> str:
        result = []

        if self.__expr:
            for var, coeff in self.__expr.items():
                result.append("+ " if coeff >= 0 else "- ")
                result.append(str(abs(coeff)) if abs(coeff) != 1 else "")
                result.append("{var} ".format(**locals()))

        if self.__sense:
            result.append(self.__sense + "= ")
            result.append(str(abs(self.__const)) if self.__const < 0 else
                          "- " + str(abs(self.__const)))
        elif self.__const != 0:
            result.append(
                "+ " + str(abs(self.__const)) if self.__const > 0
                else "- " + str(abs(self.__const)))

        return "".join(result)

    def __eq__(self, other) -> "LinExpr":
        result = self - other
        result.__sense = "="
        return result

    def __le__(self, other) -> "LinExpr":
        result = self - other
        result.__sense = "<"
        return result

    def __ge__(self, other) -> "LinExpr":
        result = self - other
        result.__sense = ">"
        return result

    def add_const(self, __const: float):
        """adds a constant value to the linear expression, in the case of
        a constraint this correspond to the right-hand-side"""
        self.__const += __const

    def add_expr(self, __expr: "LinExpr", coeff: float = 1):
        """extends a linear expression with the contents of another"""
        self.__const += __expr.__const * coeff
        for var, coeff_var in __expr.__expr.items():
            self.add_var(var, coeff_var * coeff)

    def add_term(self, __expr, coeff: float = 1):
        """extends a linear expression with another multiplied by a constant
        value coefficient"""
        if isinstance(__expr, Var):
            self.add_var(__expr, coeff)
        elif isinstance(__expr, LinExpr):
            self.add_expr(__expr, coeff)
        elif isinstance(__expr, float) or isinstance(__expr, int):
            self.add_const(__expr)

    def add_var(self, var: "Var", coeff: float = 1):
        """adds a variable with a coefficient to the constraint"""
        if var in self.__expr:
            if -EPS <= self.__expr[var] + coeff <= EPS:
                del self.__expr[var]
            else:
                self.__expr[var] += coeff
        else:
            self.__expr[var] = coeff

    def copy(self) -> "LinExpr":
        copy = LinExpr()
        copy.__const = self.__const
        copy.__expr = self.__expr.copy()
        copy.__sense = self.__sense
        return copy

    def equals(self: "LinExpr", other: "LinExpr") -> bool:
        """returns true if a linear expression equals to another,
        false otherwise"""
        if self.__sense != other.__sense:
            return False
        if len(self.__expr) != len(other.__expr):
            return False
        if abs(self.__const-other.__const) >= 1e-12:
            return False
        for (v, c) in self.__expr.items():
            if v not in self.__expr:
                return False
            oc = self.__expr[v]
            if abs(c - oc) > 1e-12:
                return False
        return True

    def __hash__(self):
        hash_el = [v.idx for v in self.__expr.keys()]
        for c in self.__expr.values():
            hash_el.append(c)
        hash_el.append(self.__const)
        hash_el.append(self.__sense)
        return hash(tuple(hash_el))

    @property
    def const(self) -> float:
        """constant part of the linear expression"""
        return self.__const

    @property
    def expr(self) -> dict:
        """the non-constant part of the linear expression

        Dictionary with pairs: (variable, coefficient) where coefficient
        is a float.
        """
        return self.__expr

    @property
    def sense(self) -> str:
        """sense of the linear expression

        sense can be EQUAL("="), LESS_OR_EQUAL("<"), GREATER_OR_EQUAL(">") or
        empty ("") if this is an affine expression, such as the objective
        function
        """
        return self.__sense


class Model:
    """ Mixed Integer Programming Model

    This is the main class, providing methods for building, optimizing,
    querying optimization results and re-optimizing Mixed-Integer Programming
    Models.

    To check how models are created please see the examples included.

    """

    def __init__(self, name: str = "",
                 sense: str = MINIMIZE,
                 solver_name: str = ""):
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
            solver_name: gurobi or cbc, searches for which
                solver is available if not informed

        """
        # initializing variables with default values
        self.name = name
        self.solver_name = solver_name
        self.solver = None

        # reading solver_name from an environment variable (if applicable)
        if not self.solver_name and "solver_name" in environ:
            self.solver_name = environ["solver_name"]
        if not self.solver_name and "solver_name".upper() in environ:
            self.solver_name = environ["solver_name".upper()]

        # creating a solver instance
        if self.solver_name.upper() == GUROBI:
            from mip.gurobi import SolverGurobi
            self.solver = SolverGurobi(self, self.name, sense)
        elif self.solver_name.upper() == CBC:
            from mip.cbc import SolverCbc
            self.solver = SolverCbc(self, self.name, sense)
        else:
            # checking which solvers are available
            from mip import gurobi
            if gurobi.has_gurobi:
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
        self.__start = None
        self.__status = OptimizationStatus.LOADED
        self.__threads = 0
        self.__n_cols = 0
        self.__n_rows = 0

    def __del__(self):
        if self.solver:
            del self.solver

    def __iadd__(self, other) -> "Model":
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

        return self

    def add_var(self, name: str = "",
                lb: float = 0.0,
                ub: float = INF,
                obj: float = 0.0,
                var_type: str = CONTINUOUS,
                column: "Column" = None) -> "Var":
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
        if var_type == BINARY:
            lb = 0.0
            ub = 1.0
        if len(name.strip()) == 0:
            nc = self.solver.num_cols()
            name = "C{:011d}".format(nc)
        self.solver.add_var(obj, lb, ub, var_type, column, name)
        self.__n_cols += 1
        return Var(self, self.__n_cols-1)

    def add_constr(self, lin_expr: "LinExpr", name: str = "") -> "Constr":
        """ Creates a new constraint (row)

        Adds a new constraint to the model, returning its reference

        Args:
            lin_expr (LinExpr): linear expression
            name (str): optional constraint name, used when saving model to\
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
            raise InvalidLinExpr("A boolean (true/false) cannot be \
            used as a constraint.")
        self.__n_rows += 1
        self.solver.add_constr(lin_expr, name)
        return Constr(self, self.__n_rows-1)

    def clear(self):
        """Clears the model

        All variables, constraints and parameters will be reset. In addition,
        a new solver instance will be instantiated to implement the
        formulation.
        """
        # creating a new solver instance
        sense = self.sense

        self.__n_cols = 0
        self.__n_rows = 0

        if self.solver_name.upper() == GUROBI:
            from mip.gurobi import SolverGurobi
            self.solver = SolverGurobi(self, self.name, sense)
        elif self.solver_name.upper() == CBC:
            from mip.cbc import SolverCbc
            self.solver = SolverCbc(self, self.name, sense)
        else:
            # checking which solvers are available
            from mip import gurobi
            if gurobi.has_gurobi:
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
        self.__start = []
        self.__status = OptimizationStatus.LOADED
        self.__threads = 0

    def copy(self, solver_name: str = None) -> "Model":
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
            copy.add_var(name=v.name, lb=v.lb, ub=v.ub, obj=v.obj,
                         var_type=v.var_type)

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

    def constr_by_name(self, name: str) -> "Constr":
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

    def var_by_name(self, name: str) -> "Var":
        """Searchers a variable by its name

        Returns:
            Variable or None if not found
        """
        v = self.solver.var_get_index(name)
        if v < 0 or v > len(self.vars):
            return None
        return self.vars[v]

    def optimize(self,
                 max_seconds: float = inf,
                 max_nodes: int = inf,
                 max_solutions: int = inf) -> OptimizationStatus:
        """ Optimizes current model

        Optimizes current model, optionally specifying processing limits.

        To optimize model :code:`m` within a processing time limit of
        300 seconds::

            m.optimize(max_seconds=300)

        Args:
            max_seconds (float): Maximum runtime in seconds (default: inf)
            max_nodes (float): Maximum number of nodes (default: inf)
            max_solutions (float): Maximum number of solutions (default: inf)

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
        if self.__threads != 0:
            self.solver.set_num_threads(self.__threads)
        # self.solver.set_callbacks(branch_selector,
        # incumbent_updater, lazy_constrs_generator)
        self.solver.set_processing_limits(max_seconds,
                                          max_nodes, max_solutions)

        self.__status = self.solver.optimize()

        return self.__status

    def read(self, path: str):
        """Reads a MIP model in :code:`.lp` or :code:`.mps` format.

        Note: all variables, constraints and parameters from the current model
        will be cleared.

        Args:
            path(str): file name
        """
        self.clear()
        self.solver.read(path)
        self.__n_cols = self.solver.num_cols()
        self.__n_rows = self.solver.num_rows()

    def relax(self):
        """ Relax integrality constraints of variables

        Changes the type of all integer and binary variables to
        continuous. Bounds are preserved.
        """
        self.solver.relax()
        for v in self.vars:
            if v.type == BINARY or v.type == INTEGER:
                v.type = CONTINUOUS

    def write(self, path: str):
        """Saves the MIP model, using the extension :code:`.lp` or
        :code:`.mps` to specify the file format.

        Args:
            path(str): file name
        """
        self.solver.write(path)

    @property
    def objective_bound(self) -> float:
        return self.solver.get_objective_bound()

    @property
    def objective(self) -> LinExpr:
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
    def objective(self, objective):
        if isinstance(objective, int) or isinstance(objective, float):
            self.solver.set_objective(LinExpr([], [], objective))
        elif isinstance(objective, Var):
            self.solver.set_objective(LinExpr([objective], [1]))
        elif isinstance(objective, LinExpr):
            self.solver.set_objective(objective)

    @property
    def verbose(self) -> int:
        """0 to disable solver messages printed on the screen, 1 to enable
        """
        return self.solver.get_verbose()

    @verbose.setter
    def verbose(self, verbose: int):
        self.solver.set_verbose(verbose)

    @property
    def threads(self) -> int:
        """number of threads to be used when solving the problem.
        0 uses solver default configuration, -1 uses the number of available
        processing cores and :math:`\geq 1` uses the specified number of
        threads. An increased number of threads may improve the solution
        time but also increases the memory consumption."""
        return self.__threads

    @threads.setter
    def threads(self, threads: int):
        self.__threads = threads

    @property
    def sense(self) -> str:
        """ The optimization sense

        Returns:
            the objective function sense, MINIMIZE (default) or (MAXIMIZE)
        """

        return self.solver.get_objective_sense()

    @sense.setter
    def sense(self, sense: str):
        self.solver.set_objective_sense(sense)

    @property
    def objective_const(self) -> float:
        """Returns the constant part of the objective function
        """
        return self.solver.get_objective_const()

    @objective_const.setter
    def objective_const(self, objective_const: float):
        self.solver.set_objective_const(objective_const)

    @property
    def objective_value(self) -> float:
        """Objective function value of the solution found
        """
        return self.solver.get_objective_value()

    @property
    def num_solutions(self) -> int:
        """Number of solutions found during the MIP search

        Returns:
            number of solutions stored in the solution pool

        """
        return self.solver.get_num_solutions()

    @property
    def objective_values(self) -> List[float]:
        """List of costs of all solutions in the solution pool

        Returns:
            costs of all solutions stored in the solution pool
            as an array from 0 (the best solution) to
            :attr:`~mip.model.model.num_solutions`-1.
        """
        return [float(self.solver.get_objective_value_i(i))
                for i in range(self.num_solutions)]

    @property
    def cuts_generator(self) -> "CutsGenerator":
        """Cut generator callback. Cut generators are called whenever a
        solution where one or more integer variables appear with
        continuous values. A cut generator will try to produce
        one or more inequalities to remove this fractional point.
        """
        return self.__cuts_generator

    @cuts_generator.setter
    def cuts_generator(self, cuts_generator: "CutsGenerator"):
        self.__cuts_generator = cuts_generator

    @property
    def lazy_constrs_generator(self) -> "LazyConstrsGenerator":
        return self.__lazy_constrs_generator

    @lazy_constrs_generator.setter
    def lazy_constrs_generator(self,
                               lazy_constrs_generator: "LazyConstrsGenerator"):
        self.__lazy_constrs_generator = lazy_constrs_generator

    @property
    def emphasis(self) -> SearchEmphasis:
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
    def emphasis(self, emphasis: SearchEmphasis):
        self.solver.set_emphasis(emphasis)

    @property
    def cuts(self) -> int:
        """controls the generation of cutting planes, 0 disables completely,
        1 (default) generates cutting planes in a moderate way,
        2 generates cutting planes aggressively and
        3 generates
        even more cutting planes. Cutting planes usually improve the LP
        relaxation bound but also make the solution time of the LP relaxation
        larger, so the overall effect is hard to predict and experimenting
        different values for this parameter may be beneficial.
        """
        return self.__cuts

    @cuts.setter
    def cuts(self, cuts: int):
        if cuts < 0 or cuts > 3:
            print('Warning: invalid value ({}) for parameter cuts, \
                  keeping old setting.'.format(self.__cuts))
        self.__cuts = cuts

    @property
    def start(self) -> List[Tuple["Var", float]]:
        """Initial feasible solution

        Enters an initial feasible solution. Only the main binary/integer
        decision variables which appear with non-zero values in the initial
        feasible solution need to be informed. Auxiliary or continuous
        variables are automatically computed.
        """
        return self.__start

    @start.setter
    def start(self, start: List[Tuple["Var", float]]):
        self.__start = start
        self.solver.set_start(start)

    @property
    def num_cols(self) -> int:
        """number of columns (variables) in the model"""
        return self.__n_cols

    @property
    def num_int(self) -> int:
        """number of integer variables in the model"""
        return self.solver.num_int()

    @property
    def num_rows(self) -> int:
        """number of rows (constraints) in the model"""
        return self.__n_rows

    @property
    def num_nz(self) -> int:
        """number of non-zeros in the constraint matrix"""
        return self.solver.num_nz()

    @property
    def cutoff(self) -> float:
        """upper limit for the solution cost, solutions with cost > cutoff
        will be removed from the search space, a small cutoff value may
        significantly speedup the search, but if cutoff is set to a value too
        low the model will become infeasible"""
        return self.solver.get_cutoff()

    @cutoff.setter
    def cutoff(self, cutoff: float):
        self.solver.set_cutoff(cutoff)

    @property
    def max_mip_gap_abs(self) -> float:
        """tolerance for the quality of the optimal solution, if a
        solution with cost :math:`c` and a lower bound :math:`l` are available
        and :math:`c-l<` :code:`mip_gap_abs`,
        the search will be concluded, see mip_gap to determine
        a percentage value """
        return self.solver.get_mip_gap_abs()

    @max_mip_gap_abs.setter
    def max_mip_gap_abs(self, max_mip_gap_abs: float):
        self.solver.set_mip_gap(max_mip_gap_abs)

    @property
    def max_mip_gap(self) -> float:
        """value indicating the tolerance for the maximum percentage deviation
        from the optimal solution cost, if a solution with cost :math:`c` and
        a lower bound :math:`l` are available and
        :math:`(c-l)/l <` :code:`max_mip_gap` the search will be concluded."""
        return self.solver.get_mip_gap()

    @max_mip_gap.setter
    def max_mip_gap(self, max_mip_gap: float):
        self.solver.set_mip_gap(max_mip_gap)

    @property
    def max_seconds(self) -> float:
        """time limit in seconds for search"""
        return self.solver.get_max_seconds()

    @max_seconds.setter
    def max_seconds(self, max_seconds: float):
        self.solver.set_max_seconds(max_seconds)

    @property
    def max_nodes(self) -> int:
        """maximum number of nodes to be explored in the search tree"""
        return self.solver.get_max_nodes()

    @max_nodes.setter
    def max_nodes(self, max_nodes: int):
        self.solver.set_max_nodes(max_nodes)

    @property
    def max_solutions(self) -> int:
        """solution limit, search will be stopped when :code:`max_solutions`
        were found"""
        return self.solver.get_max_solutions()

    @max_solutions.setter
    def max_solutions(self, max_solutions: int):
        self.solver.set_max_solutions(max_solutions)

    @property
    def status(self) -> OptimizationStatus:
        """ optimization status, which can be OPTIMAL(0), ERROR(-1),
        INFEASIBLE(1), UNBOUNDED(2). When optimizing problems
        with integer variables some additional cases may happen, FEASIBLE(3)
        for the case when a feasible solution was found but optimality was
        not proved, INT_INFEASIBLE(4) for the case when the lp relaxation is
        feasible but no feasible integer solution exists and
        NO_SOLUTION_FOUND(5) for the case when an integer solution was not
        found in the optimization.
        """
        return self.__status

    def remove(self, objects):
        """removes variable(s) and/or constraint(s) from the model

        Args:
            objects: can be a Var, a Constr or a list of these objects
        """
        if isinstance(objects, Var):
            self.solver.remove_vars([objects.idx])
        elif isinstance(objects, Constr):
            self.solver.remove_constrs([objects.idx])
        elif isinstance(objects, list):
            vlist = []
            clist = []
            for o in objects:
                if isinstance(o, Var):
                    vlist.append(o.idx)
                elif isinstance(o, Constr):
                    clist.append(o.idx)
                else:
                    raise Exception("Cannot handle removal of object of type "
                                    + type(o) + " from model.")
            if vlist:
                vlist.sort()
                self.solver.remove_vars(vlist)
                self.__n_cols -= len(vlist)
            if clist:
                clist.sort()
                self.solver.remove_constrs(clist)
                self.__n_rows -= len(clist)


class Solver:

    def __init__(self, model: Model, name: str, sense: str):
        self.model = model
        self.name = name
        self.sense = sense

    def __del__(self): pass

    def add_var(self,
                name: str = "",
                obj: float = 0,
                lb: float = 0,
                ub: float = INF,
                var_type: str = CONTINUOUS,
                column: "Column" = None):
        pass

    def add_constr(self, lin_expr: "LinExpr", name: str = ""):
        pass

    def get_objective_bound(self) -> float: pass

    def get_objective(self) -> LinExpr: pass

    def get_objective_const(self) -> float: pass

    def relax(self): pass

    def optimize(self) -> OptimizationStatus: pass

    def get_objective_value(self) -> float: pass

    def get_objective_value_i(self, i: int) -> float: pass

    def get_num_solutions(self) -> int: pass

    def get_objective_sense(self) -> str: pass

    def set_objective_sense(self, sense: str): pass

    def set_start(self, start: List[Tuple["Var", float]]): pass

    def set_objective(self, lin_expr: "LinExpr", sense: str = ""): pass

    def set_objective_const(self, const: float): pass

    def set_callbacks(self,
                      branch_selector: "BranchSelector" = None,
                      incumbent_updater: "IncumbentUpdater" = None,
                      lazy_constrs_generator: "LazyConstrsGenerator" = None):
        pass

    def set_processing_limits(self,
                              max_time: float = inf,
                              max_nodes: int = inf,
                              max_sol: int = inf):
        pass

    def get_max_seconds(self) -> float: pass

    def set_max_seconds(self, max_seconds: float): pass

    def get_max_solutions(self) -> int: pass

    def set_max_solutions(self, max_solutions: int): pass

    def get_max_nodes(self) -> int: pass

    def set_max_nodes(self, max_nodes: int): pass

    def set_num_threads(self, threads: int): pass

    def write(self, file_path: str): pass

    def read(self, file_path: str): pass

    def num_cols(self) -> int: pass

    def num_rows(self) -> int: pass

    def num_nz(self) -> int: pass

    def num_int(self) -> int: pass

    def get_emphasis(self) -> SearchEmphasis: pass

    def set_emphasis(self, emph: SearchEmphasis): pass

    def get_cutoff(self) -> float: pass

    def set_cutoff(self, cutoff: float): pass

    def get_mip_gap_abs(self) -> float: pass

    def set_mip_gap_abs(self, mip_gap_abs: float): pass

    def get_mip_gap(self) -> float: pass

    def set_mip_gap(self, mip_gap: float): pass

    def get_verbose(self) -> int: pass

    def set_verbose(self, verbose: int): pass

    # Constraint-related getters/setters

    def constr_get_expr(self, constr: Constr) -> LinExpr: pass

    def constr_set_expr(self, constr: Constr, value: LinExpr) -> LinExpr: pass

    def constr_get_name(self, idx: int) -> str: pass

    def constr_get_pi(self, constr: Constr) -> float: pass

    def remove_constrs(self, constrsList: List[int]): pass

    def constr_get_index(self, name: str) -> int: pass

    # Variable-related getters/setters

    def var_get_lb(self, var: "Var") -> float: pass

    def var_set_lb(self, var: "Var", value: float): pass

    def var_get_ub(self, var: "Var") -> float: pass

    def var_set_ub(self, var: "Var", value: float): pass

    def var_get_obj(self, var: "Var") -> float: pass

    def var_set_obj(self, var: "Var", value: float): pass

    def var_get_var_type(self, var: "Var") -> str: pass

    def var_set_var_type(self, var: "Var", value: str): pass

    def var_get_column(self, var: "Var") -> Column: pass

    def var_set_column(self, var: "Var", value: Column): pass

    def var_get_rc(self, var: "Var") -> float: pass

    def var_get_x(self, var: "Var") -> float: pass

    def var_get_xi(self, var: "Var", i: int) -> float: pass

    def var_get_name(self, idx: int) -> str: pass

    def remove_vars(self, varsList: List[int]): pass

    def var_get_index(self, name: str) -> int: pass


class Var:
    """
    Objects of class Var are decision variables of a model. The creation
    of variables is performed calling the :meth:`~mip.model.Model.add_var`
    method of the Model class.

    """

    def __init__(self,
                 model: Model,
                 idx: int):
        self.__model = model
        self.idx = idx

    def __hash__(self) -> int:
        return self.idx

    def __add__(self, other) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, 1])
        elif isinstance(other, LinExpr):
            return other.__add__(self)
        elif isinstance(other, int) or isinstance(other, float):
            return LinExpr([self], [1], other)

    def __radd__(self, other) -> LinExpr:
        return self.__add__(other)

    def __sub__(self, other) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, -1])
        elif isinstance(other, LinExpr):
            return (-other).__iadd__(self)
        elif isinstance(other, int) or isinstance(other, float):
            return LinExpr([self], [1], -other)

    def __rsub__(self, other) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [-1, 1])
        elif isinstance(other, LinExpr):
            return other.__sub__(self)
        elif isinstance(other, int) or isinstance(other, float):
            return LinExpr([self], [-1], other)

    def __mul__(self, other) -> LinExpr:
        assert isinstance(other, int) or isinstance(other, float)
        return LinExpr([self], [other])

    def __rmul__(self, other) -> LinExpr:
        return self.__mul__(other)

    def __truediv__(self, other) -> LinExpr:
        assert isinstance(other, int) or isinstance(other, float)
        return self.__mul__(1.0 / other)

    def __neg__(self) -> LinExpr:
        return LinExpr([self], [-1.0])

    def __eq__(self, other) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, -1], sense="=")
        elif isinstance(other, LinExpr):
            return other == self
        elif isinstance(other, int) or isinstance(other, float):
            if other != 0:
                return LinExpr([self], [1], -1 * other, sense="=")
            return LinExpr([self], [1], sense="=")

    def __le__(self, other) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, -1], sense="<")
        elif isinstance(other, LinExpr):
            return other >= self
        elif isinstance(other, int) or isinstance(other, float):
            if other != 0:
                return LinExpr([self], [1], -1 * other, sense="<")
            return LinExpr([self], [1], sense="<")

    def __ge__(self, other) -> LinExpr:
        if isinstance(other, Var):
            return LinExpr([self, other], [1, -1], sense=">")
        elif isinstance(other, LinExpr):
            return other <= self
        elif isinstance(other, int) or isinstance(other, float):
            if other != 0:
                return LinExpr([self], [1], -1 * other, sense=">")
            return LinExpr([self], [1], sense=">")

    @property
    def name(self) -> str:
        """variable name"""
        return self.__model.solver.var_get_name(self.idx)

    def __str__(self) -> str:
        return self.name

    @property
    def lb(self) -> float:
        """the variable lower bound"""
        return self.__model.solver.var_get_lb(self)

    @lb.setter
    def lb(self, value: float):
        self.__model.solver.var_set_lb(self, value)

    @property
    def ub(self) -> float:
        """the variable upper bound"""
        return self.__model.solver.var_get_ub(self)

    @ub.setter
    def ub(self, value: float):
        self.__model.solver.var_set_ub(self, value)

    @property
    def obj(self) -> float:
        """coefficient of a variable in the objective function"""
        return self.__model.solver.var_get_obj(self)

    @obj.setter
    def obj(self, value: float):
        self.__model.solver.var_set_obj(self, value)

    @property
    def var_type(self) -> str:
        """variable type ('B') BINARY, ('C') CONTINUOUS and ('I') INTEGER"""
        return self.__model.solver.var_get_var_type(self)

    @var_type.setter
    def var_type(self, value: str):
        assert value in (BINARY, CONTINUOUS, INTEGER)
        self.__model.solver.var_set_var_type(self, value)

    @property
    def column(self) -> Column:
        """coefficients of variable in constraints"""
        return self.__model.solver.var_get_column(self)

    @column.setter
    def column(self, value: Column):
        self.__model.solver.var_set_column(self, value)

    @property
    def rc(self) -> float:
        """reduced cost, only available after a linear programming __model (no
        integer variables) is optimized"""
        if self.__model.status != OptimizationStatus.OPTIMAL:
            raise SolutionNotAvailable('Solution not available.')

        return self.__model.solver.var_get_rc(self)

    @property
    def x(self) -> float:
        """solution value"""
        if self.__model.status == OptimizationStatus.LOADED:
            raise SolutionNotAvailable('Model was not optimized, \
                solution not available.')
        elif (self.__model.status == OptimizationStatus.INFEASIBLE
              or self.__model.status == OptimizationStatus.CUTOFF):
            raise SolutionNotAvailable('Infeasible __model, \
                solution not available.')
        elif self.__model.status == OptimizationStatus.UNBOUNDED:
            raise SolutionNotAvailable('Unbounded __model, solution not \
                available.')
        elif self.__model.status == OptimizationStatus.NO_SOLUTION_FOUND:
            raise SolutionNotAvailable('Solution not found \
                during optimization.')

        return self.__model.solver.var_get_x(self)

    def xi(self, i: int) -> float:
        """solution value for this variable in the :math:`i`-th solution from
        the solution pool"""
        if self.__model.status == OptimizationStatus.LOADED:
            raise SolutionNotAvailable('Model was not optimized, \
                solution not available.')
        elif (self.__model.status == OptimizationStatus.INFEASIBLE or
              self.__model.status == OptimizationStatus.CUTOFF):
            raise SolutionNotAvailable('Infeasible __model, \
                solution not available.')
        elif self.__model.status == OptimizationStatus.UNBOUNDED:
            raise SolutionNotAvailable('Unbounded __model, \
                solution not available.')
        elif self.__model.status == OptimizationStatus.NO_SOLUTION_FOUND:
            raise SolutionNotAvailable('Solution not found \
                during optimization.')

        return self.__model.solver.var_get_xi(self, i)


class VarList(Sequence):
    def __init__(self, model: Model):
        self.__model = model

    def __getitem__(self, index: int) -> Var:
        if index >= len(self) or index < -len(self):
            raise IndexError
        if index < 0:
            index = len(self)+index
        assert index >= 0 and index < len(self)
        return Var(self.__model, index)

    def __len__(self) -> int:
        return self.__model.solver.num_cols()


class ConstrList(Sequence):
    def __init__(self, model: Model):
        self.__model = model

    def __getitem__(self, index: int) -> Constr:
        if index >= len(self) or index < -len(self):
            raise IndexError
        if index < 0:
            index = len(self)+index
        assert index >= 0 and index < len(self)
        return Constr(self.__model, index)

    def __len__(self) -> int:
        return self.__model.solver.num_rows()


class BranchSelector:
    def __init__(self, model: Model):
        self.model = model

    def select_branch(self,
                      relax_solution: List[Tuple[Var, float]]
                      ) -> Tuple[Var, int]:
        raise NotImplementedError()


class LazyConstrsGenerator:
    def __init(self, model: Model):
        self.model = model

    def generate_lazy_constrs(self, solution: List[Tuple[Var, float]]
                              ) -> List[LinExpr]:
        raise NotImplementedError()


def xsum(terms) -> LinExpr:
    result = LinExpr()
    for term in terms:
        result.add_term(term)
    return result


# function aliases
quicksum = xsum


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
                        customCbcLib = cols[1].\
                            lstrip().rstrip().replace('"', "")


print("Using Python-MIP package version {}".format(VERSION))
customCbcLib = ""
read_custom_settings()

# vim: ts=4 sw=4 et
