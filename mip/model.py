"""MIP Model

This module implements abstractions for working with Mixed-Integer Programming
Models.

"""

from typing import Dict, List, Tuple

from mip.constants import *
from math import inf


class Column:
    """ A column (variable) in the constraint matrix

        To create a column see Model.add_var

    """
    def __init__(self, constrs: List["Constr"] = None, coeffs: List[float] = None):
        self.constrs = constrs if constrs else list()
        self.coeffs = coeffs if coeffs else list()


class Constr:
    """ A row (constraint) in the constraint matrix

        a constraint can be added to the model using the overloaded operator
        +=, e.g., if m is a model:

        m += 3*x1 + 4*x2 <= 5

        summation expressions are also supported:

        m += xsum(x[i] for i in range(n)) == 1

    """

    def __init__(self, model: "Model", idx: int, name: str = ""):
        self.model = model
        self.idx = idx
        self.name = name  # discuss this var

    def __hash__(self) -> int:
        return self.idx

    def __str__(self) -> str:
        return self.name

    @property
    def pi(self) -> float:
        return self.model.solver.constr_get_pi(self)

    @property
    def expr(self) -> "LinExpr":
        return self.model.solver.constr_get_expr(self)

    @expr.setter
    def expr(self, value: "LinExpr") -> None:
        self.model.solver.constr_set_expr(self, value)


class LinExpr:
    """ A Linear Expression

    Linear expressions are used to enter the objective function and constraints of the model.

    Consider a model object m, the objective function of m can be specified as:

    m += 10*x1 + 7*x4

    summation can also be used:

    m += sum(3*x[i] i in range(n)) - sum(x[i] i in range(m))

    If not specified in the construction of the model object, it is assumed
    that the model is a minimization one.

    A constraint is just a linear expression with the addition of a sense (==,
    <= or >=) and a right hand side, e.g.:

    m += x1 + x2 + x3 == 1

    """

    def __init__(self,
                 variables: List["Var"] = None,
                 coeffs: List[float] = None,
                 const: float = 0,
                 sense: str = ""):
        self.const = const
        self.expr  = {}
        self.sense = sense

        if variables:
            assert len(variables) == len(coeffs)
            for i in range(len(coeffs)):
                if coeffs[i] == 0:
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
        assert isinstance(other, int) or isinstance(other, float)
        result = self.copy()
        result.const *= other
        for var in result.expr.keys():
            result.expr[var] *= other

        # if constraint sense will change
        if self.sense == GREATER_OR_EQUAL and other <= -1e-8:
            self.sense = LESS_OR_EQUAL
        if self.sense == LESS_OR_EQUAL and other <= -1e-8:
            self.sense = GREATER_OR_EQUAL

        return result

    def __rmul__(self, other) -> "LinExpr":
        return self.__mul__(other)

    def __imul__(self, other) -> "LinExpr":
        assert isinstance(other, int) or isinstance(other, float)
        self.const *= other
        for var in self.expr.keys():
            self.expr[var] *= other
        return self

    def __truediv__(self, other) -> "LinExpr":
        assert isinstance(other, int) or isinstance(other, float)
        result = self.copy()
        result.const /= other
        for var in result.expr.keys():
            result.expr[var] /= other
        return result

    def __itruediv__(self, other) -> "LinExpr":
        assert isinstance(other, int) or isinstance(other, float)
        self.const /= other
        for var in self.expr.keys():
            self.expr[var] /= other
        return self

    def __neg__(self) -> "LinExpr":
        return self.__mul__(-1)

    def __str__(self) -> str:
        result = []

        if self.expr:
            for var, coeff in self.expr.items():
                result.append("+ " if coeff >= 0 else "- ")
                result.append(str(abs(coeff)) if abs(coeff) != 1 else "")
                result.append("{var} ".format(**locals()))

        if self.sense:
            result.append(self.sense + "= ")
            result.append(str(abs(self.const)) if self.const < 0 else "- " + str(abs(self.const)))
        elif self.const != 0:
            result.append(
                "+ " + str(abs(self.const)) if self.const > 0 else "- " + str(abs(self.const)))

        return "".join(result)

    def __eq__(self, other) -> "LinExpr":
        result = self - other
        result.sense = "="
        return result

    def __le__(self, other) -> "LinExpr":
        result = self - other
        result.sense = "<"
        return result

    def __ge__(self, other) -> "LinExpr":
        result = self - other
        result.sense = ">"
        return result

    def add_const(self, const: float) -> None:
        self.const += const

    def add_expr(self, expr: "LinExpr", coeff: float = 1) -> None:
        self.const += expr.const * coeff
        for var, coeff_var in expr.expr.items():
            self.add_var(var, coeff_var * coeff)

    def add_term(self, expr, coeff: float = 1) -> None:
        if isinstance(expr, Var):
            self.add_var(expr, coeff)
        elif isinstance(expr, LinExpr):
            self.add_expr(expr, coeff)
        elif isinstance(expr, float) or isinstance(expr, int):
            self.add_const(expr)

    def add_var(self, var: "Var", coeff: float = 1) -> None:
        if var in self.expr:
            if -EPS <= self.expr[var] + coeff <= EPS:
                del self.expr[var]
            else:
                self.expr[var] += coeff
        else:
            self.expr[var] = coeff

    def copy(self) -> "LinExpr":
        copy = LinExpr()
        copy.const = self.const
        copy.expr = self.expr.copy()
        copy.sense = self.sense
        return copy


class Model:
    """ Mixed Integer Programming Model

    This is the main class, providing methods for building, optimizing,
    querying optimization results and reoptimizing Mixed-Integer Programming
    Models.

    To check how models are created please see the examples included.

    """

    def __init__(self, name: str = "",
                 sense: str = MINIMIZE,
                 solver_name: str = ''):
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
        self.sense = sense
        self.solver_name = solver_name
        self.solver = None

        # list of constraints and variables
        self.constrs = []
        self.constrs_dict = {}
        self.vars = []
        self.vars_dict = {}

        if solver_name.upper() == GUROBI:
            from mip.gurobi import SolverGurobi
            self.solver = SolverGurobi(self, name, sense)
        elif solver_name.upper() == CBC:
            from mip.cbc import SolverCbc
            self.solver = SolverCbc(self, name, sense)
        else:
            # checking which solvers are available
            from mip import gurobi
            from mip import cbc

            # search for the best solver available
            if gurobi.has_gurobi:
                from mip.gurobi import SolverGurobi
                self.solver = SolverGurobi(self, name, sense)
                self.solver_name = "gurobi"
            elif cbc.has_cbc:
                from mip.cbc import SolverCbc
                self.solver = SolverCbc(self, name, sense)
                self.solver_name = "cbc"

    def __del__(self):
        if self.solver:
            del self.solver

    def __iadd__(self, other) -> 'Model':
        if isinstance(other, LinExpr):
            if len(other.sense) == 0:
                # adding objective function components
                self.set_objective(other)
            else:
                # adding constraint
                self.add_constr(other)
        elif isinstance(other, tuple):
            if isinstance(other[0], LinExpr) and isinstance(other[1], str):
                if len(other[0].sense) == 0:
                    self.set_objective(other[0])
                else:
                    self.add_constr(other[0], other[1])

        return self

    def add_var(self, name: str = "",
                lb: float = 0.0,
                ub: float = INF,
                obj: float = 0.0,
                type: str = CONTINUOUS,
                column: "Column" = None) -> "Var":
        """ Creates a new variable

        Adds a new variable to the model.

        Args:
            name (str): variable name (optional)
            lb (float): variable lower bound, default 0.0
            ub (float): variable upper bound, default infinity
            obj (float): coefficient of this variable in the objective function, default 0
            type (str): CONTINUOUS ("C"), BINARY ("B") or INTEGER ("I")
            column (Column): constraints where this variable will appear, necessary \
            only when constraints are already created in the model and a new \
            variable will be created.

        Examples:

            To add a variable x which is continuous and greater or equal to zero to model m::

                x = m.add_var()

            The following code creates a vector of binary variables x[0], ..., x[n-1] to model m::

                x = [m.add_var(type=BINARY) for i in range(n)]


        """
        if type == BINARY:
            lb = 0.0
            ub = 1.0
        if len(name.strip()) == 0:
            nc = self.solver.num_cols()
            name = 'C{:011d}'.format(nc)
        idx = self.solver.add_var(obj, lb, ub, type, column, name)
        self.vars.append(Var(self, idx, name))
        self.vars_dict[name] = self.vars[-1]
        return self.vars[-1]

    def add_constr(self, lin_expr: "LinExpr", name: str = "") -> Constr:
        """ Creates a new constraint (row)

        Adds a new constraint to the model

        Args:
            lin_expr (LinExpr): linear expression
            name (str): optional constraint name, used when saving model to lp or mps files

        Examples:

        The following code adds the constraint :math:`x_1 + x_2 \leq 1`::

            m += x1 + x2 <= 1

        Which is equivalent to::

            m.add_constr( x1 + x2 <= 1 )

        Summation expressions can be used also, to add the constraint :math:`\displaystyle \sum_{i=0}^{n-1} x_i = y` and name this constraint cons1::

            m += xsum(x[i] for i in range(n)) == y, 'cons1'

        """

        if isinstance(lin_expr, bool):
            return None  # empty constraint
        idx = self.solver.add_constr(lin_expr, name)
        self.constrs.append(Constr(self, idx, name))
        self.constrs_dict[name] = self.constrs[-1]
        return self.constrs[-1]

    def copy(self, solver_name: str = None) -> "Model":
        """ Creates a copy of the current model

        Args:
            solver_name(str): solver name (optional)

        Returns:
            Model: clone of current model

        """
        if not solver_name:
            solver_name = self.solver_name
        copy = Model(self.name, self.sense, solver_name)

        # adding variables
        for v in self.vars:
            copy.add_var(name=v.name, lb=v.lb, ub=v.ub, obj=v.obj, type=v.type)

        # adding constraints
        for c in self.constrs:
            expr = c.expr  # todo: make copy of constraint's lin_expr
            copy.add_constr(lin_expr=expr, name=c.name)

        # setting objective function's constant
        copy.set_objective_const(self.get_objective_const())

        return copy

    def get_constr_by_name(self, name) -> "Constr":
        return self.constrs_dict.get(name, None)

    def get_objective(self) -> LinExpr:
        """ Returns the objective function

        Returns:
            LinExpr: the model objective function

        """
        return self.solver.get_objective()

    def get_objective_const(self) -> float:
        return self.solver.get_objective_const()

    def get_objective_value(self) -> float:
        """ Objective function value

        Returns:
            float: returns the objetive function value of the solution found.

        """
        return self.solver.get_objective_value()

    def get_num_solutions(self) -> int:
        """ Number of solutions found during the MIP search

        Returns:
            int: number of solutions stored in the solution pool

        """
        return self.solver.get_num_solutions()

    def get_objective_value_i(self, i : int) -> float:
        """ Cost of the i-th solution found

        Returns:
            float: cost of the i-th best solution from the solution pool
        """
        return self.solver.get_objective_value_i(i)
        

    def get_var_by_name(self, name) -> "Var":
        """ Searchers a variable by its name

        Returns:
            Var: a reference to a variable
        """
        return self.vars_dict.get(name, None)

    def relax(self):
        """ Relax integrality constraints of variables

        Changes the type of all integer and binary variables to
        continuous. Bounds are preserved.
        """
        self.solver.relax()
        for v in self.vars:
            if v.type == BINARY or v.type == INTEGER:
                v.type = CONTINUOUS

    def optimize(self,
                 branch_selector: "BranchSelector" = None,
                 cuts_generator: "CutsGenerator" = None,
                 incumbent_updater: "IncumbentUpdater" = None,
                 lazy_constrs_generator: "LazyConstrsGenerator" = None,
                 max_seconds: float = inf,
                 max_nodes: int = inf,
                 max_solutions: int = inf) -> int:
        """ Optimizes current model

        Optimizes current model, optionally specifying processing limits.

        To optimize model m within a processing time limit of 300 seconds::

            m.optimize(max_seconds=300)

        Args:
            branch_selector (BranchSelector): Callback to select branch (an object of a class inheriting from BranchSelector must be passed)
            cuts_generator (CutsGenerator): Callback to generate cuts (an object of a class inheriting from CutsGenerator must be passed)
            incumbent_updater (IncumbentUpdater): Callback to update incumbent solution (an object of a class inheriting from IncumbentUpdater must be passed)
            lazy_constrs_generator (LazyConstrsGenerator): Callback to include lazy generated constraints (an object of a class inheriting from LazyConstrsGenerator must be passed)
            max_seconds (float): Maximum runtime in seconds (default: inf)
            max_nodes (float): Maximum number of nodes (default: inf)
            max_solutions (float): Maximum number of solutions (default: inf)

        Returns:
            int: optimization status, which can be OPTIMAL(0), ERROR(-1), INFEASIBLE(1), UNBOUNDED(2). When optimizing problems
            with integer variables some additional cases may happen, FEASIBLE(3) for the case when a feasible solution was found
            but optimality was not proved, INT_INFEASIBLE(4) for the case when the lp relaxation is feasible but no feasible integer
            solution exists and NO_SOLUTION_FOUND(5) for the case when an integer solution was not found in the optimization.

        """
        self.solver.set_callbacks(branch_selector, cuts_generator, incumbent_updater, lazy_constrs_generator)
        self.solver.set_processing_limits(max_seconds, max_nodes, max_solutions)

        return self.solver.optimize()

    def read(self, path: str) -> None:
        """ Reads a MIP model

        Reads a MIP model in .lp or .mps file format.

        Args:
            path(str): file name

        """
        self.solver.read(path)
        n_cols = self.solver.num_cols()
        n_rows = self.solver.num_rows()
        for i in range(n_cols):
            self.vars.append(Var(self, i, self.solver.var_get_name(i)))
            self.vars_dict[self.vars[-1].name] = self.vars[-1]
        for i in range(n_rows):
            self.constrs.append(Constr(self, i, self.solver.constr_get_name(i)))
            self.constrs_dict[self.constrs[-1].name] = self.constrs[-1]
        self.sense = self.solver.get_objective_sense()

    def set_start(self, variables: List["Var"], values: List[float]):
        """ Enter an initial feasible solution

        Enters an initial feasible solution. Only the main binary/integer decision variables.
        Auxiliary or continuous variables are automatically computed.

        Args:
            variables(List[Var]): list of variables 
            values(List[float]): list of variable values in initial feasible solution

        """
        self.solver.set_start(variables, values)

    def set_objective(self, expr, sense: str = "") -> None:
        """ Modifies the objective function

        Args:
            expr(LinExpr): linear expression
            sense(str): MINIMIZE("MIN") (default) or MAXIMIZE("MAX") (optional)

        Examples:

            The following code adds all x variables x[0], ..., x[n-1], with
            to the objective function of model m with weight w::

                m.set_objective(xsum(w*x[i] for i in range(n)))

            A simpler way to define the objective function is the use of the
            model operator += ::

                m += xsum(w*x[i] for i in range(n))

            Note that the only difference of adding a constraint is the lack of
            a sense and a rhs.

        """

        if isinstance(expr, int) or isinstance(expr, float):
            self.solver.set_objective(LinExpr([], [], expr))
        elif isinstance(expr, Var):
            self.solver.set_objective(LinExpr([expr], [1]))
        elif isinstance(expr, LinExpr):
            self.solver.set_objective(expr, sense)

    def set_objective_const(self, const: float) -> None:
        return self.solver.set_objective_const(const)

    def write(self, path: str) -> None:
        """ Saves the the MIP model

        Args:
            path(str): file name

        Saves the the MIP model, use the extension ".lp" or ".mps" in the file
        name to specify the file format.

        """
        self.solver.write(path)

    @property
    def num_cols(self) -> int:
        return len(self.vars)

    @property
    def num_rows(self) -> int:
        return len(self.constrs)


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
                type: str = CONTINUOUS,
                column: "Column" = None) -> int:
        if type == BINARY:
            lb = 0.0
            ub = 1.0

    def add_constr(self, lin_expr: "LinExpr", name: str = "") -> int: pass

    def get_objective(self) -> LinExpr: pass

    def get_objective_const(self) -> float: pass

    def relax(self): pass

    def optimize(self) -> int: pass

    def get_objective_value(self) -> float: pass

    def get_objective_value_i(self, i : int) -> float: pass

    def get_num_solutions(self) -> int: pass

    def get_objective_sense(self) -> str: pass

    def set_start(self, variables: List["Var"], values: List[float]) -> None: pass

    def set_objective(self, lin_expr: "LinExpr", sense: str = "") -> None: pass

    def set_objective_const(self, const: float) -> None: pass

    def set_callbacks(self,
                      branch_selector: "BranchSelector" = None,
                      cuts_generator: "CutsGenerator" = None,
                      incumbent_updater: "IncumbentUpdater" = None,
                      lazy_constrs_generator: "LazyConstrsGenerator" = None) -> None:
        pass

    def set_processing_limits(self,
                              max_time: float = inf,
                              max_nodes: int = inf,
                              max_sol: int = inf):
        pass

    def write(self, file_path: str) -> None: pass

    def read(self, file_path: str) -> None: pass

    def num_cols(self) -> int: pass

    def num_rows(self) -> int: pass

    # Constraint-related getters/setters

    def constr_get_expr(self, constr: Constr) -> LinExpr: pass

    def constr_set_expr(self, constr: Constr, value: LinExpr) -> LinExpr: pass

    def constr_get_name(self, idx: int) -> str: pass

    def constr_get_pi(self, constr: Constr) -> float: pass

    # Variable-related getters/setters

    def var_get_lb(self, var: "Var") -> float: pass

    def var_set_lb(self, var: "Var", value: float) -> None: pass

    def var_get_ub(self, var: "Var") -> float: pass

    def var_set_ub(self, var: "Var", value: float) -> None: pass

    def var_get_obj(self, var: "Var") -> float: pass

    def var_set_obj(self, var: "Var", value: float) -> None: pass

    def var_get_type(self, var: "Var") -> str: pass

    def var_set_type(self, var: "Var", value: str) -> None: pass

    def var_get_column(self, var: "Var") -> Column: pass

    def var_set_column(self, var: "Var", value: Column) -> None: pass

    def var_get_rc(self, var: "Var") -> float: pass

    def var_get_x(self, var: "Var") -> float: pass

    def var_get_xi(self, var: "Var", i: int) -> float: pass

    def var_get_name(self, idx: int) -> str: pass


class Var:

    def __init__(self,
                 model: Model,
                 idx: int,
                 name: str = ""):
        self.model = model
        self.idx = idx
        self.name = name  # discuss this var

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

    def __str__(self) -> str:
        return self.name

    @property
    def lb(self) -> float:
        return self.model.solver.var_get_lb(self)

    @lb.setter
    def lb(self, value: float) -> None:
        self.model.solver.var_set_lb(self, value)

    @property
    def ub(self) -> float:
        return self.model.solver.var_get_ub(self)

    @ub.setter
    def ub(self, value: float) -> None:
        self.model.solver.var_set_ub(self, value)

    @property
    def obj(self) -> float:
        return self.model.solver.var_get_obj(self)

    @obj.setter
    def obj(self, value: float) -> None:
        self.model.solver.var_set_obj(self, value)

    @property
    def type(self) -> str:
        return self.model.solver.var_get_type(self)

    @type.setter
    def type(self, value: str) -> None:
        assert value in (BINARY, CONTINUOUS, INTEGER)
        self.model.solver.var_set_type(self, value)

    @property
    def column(self) -> Column:
        return self.model.solver.var_get_column(self)

    @column.setter
    def column(self, value: Column) -> None:
        self.model.solver.var_set_column(self, value)

    @property
    def rc(self) -> float:
        return self.model.solver.var_get_rc(self)

    @property
    def x(self) -> float:
        return self.model.solver.var_get_x(self)

    def xi(self, i : int) -> float:
        return self.model.solver.var_get_xi(self, i)


class BranchSelector:
    def __init__(self, model: Model):
        self.model = model

    def select_branch(self, relax_solution: List[Tuple[Var, float]]) -> Tuple[Var, int]:
        raise NotImplementedError()


class CutsGenerator:
    def __init__(self, model: Model):
        self.model = model

    def generate_cuts(self, relax_solution: List[Tuple[Var, float]]) -> List[LinExpr]:
        raise NotImplementedError()


class IncumbentUpdater:
    def __init__(self, model: Model):
        self.model = model

    def update_incumbent(self, solution: List[Tuple[Var, float]]) -> List[Tuple[Var, float]]:
        raise NotImplementedError()


class LazyConstrsGenerator:
    def __init(self, model: Model):
        self.model = model

    def generate_constrs(self, solution: List[Tuple[Var, float]]) -> List[LinExpr]:
        raise NotImplementedError()


def xsum(terms) -> LinExpr:
    result = LinExpr()
    for term in terms:
        result.add_term(term)
    return result


# function aliases
quicksum = xsum

print('using python mip package version 1.0.16')

# vim: ts=4 sw=4 et
