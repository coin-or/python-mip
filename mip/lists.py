from collections.abc import Sequence
from typing import List
import numbers
import mip


class VarList(Sequence):
    """ List of model variables (:class:`~mip.Var`).

        The number of variables of a model :code:`m` can be queried as
        :code:`len(m.vars)` or as :code:`m.num_cols`.

        Specific variables can be retrieved by their indices or names.
        For example, to print the lower bounds of the first
        variable or of a varible named :code:`z`, you can use, respectively:

        .. code-block:: python

            print(m.vars[0].lb)

        .. code-block:: python

            print(m.vars['z'].lb)
    """

    def __init__(self: "VarList", model: "mip.Model"):
        self.__model = model
        self.__vars = []  # type: List[Var]

    def add(
        self,
        name: str = "",
        lb: numbers.Real = 0.0,
        ub: numbers.Real = mip.INF,
        obj: numbers.Real = 0.0,
        var_type: str = mip.CONTINUOUS,
        column: "mip.Column" = None,
    ) -> "mip.Var":
        if not name:
            name = "var({})".format(len(self.__vars))
        if var_type == mip.BINARY:
            lb = 0.0
            ub = 1.0
        new_var = mip.Var(self.__model, len(self.__vars))
        self.__model.solver.add_var(obj, lb, ub, var_type, column, name)
        self.__vars.append(new_var)
        return new_var

    def __getitem__(self: "VarList", key):
        if isinstance(key, str):
            return self.__model.var_by_name(key)
        return self.__vars[key]

    def __len__(self) -> int:
        return len(self.__vars)

    def update_vars(self: "VarList", n_vars: int):
        self.__vars = [mip.Var(self.__model, i) for i in range(n_vars)]

    def remove(self: "VarList", vars: List["mip.Var"]):
        iv = [1 for i in range(len(self.__vars))]
        vlist = [v.idx for v in vars]
        vlist.sort()
        for i in vlist:
            iv[i] = 0
        self.__model.solver.remove_vars(vlist)
        i = 0
        for v in self.__vars:
            if iv[v.idx] == 0:
                v.idx = -1
            else:
                v.idx = i
                i += 1
        self.__vars = [v for v in self.__vars if v.idx != -1]


# same as VarList but does not stores
# references for variables, used in
# callbacks
class VVarList(Sequence):
    def __init__(self: "VVarList", model: "mip.Model", start: int = -1, end: int = -1):
        self.__model = model
        if start == -1:
            self.__start = 0
            self.__end = model.solver.num_cols()
        else:
            self.__start = start
            self.__end = end

    def add(
        self: "VVarList",
        name: str = "",
        lb: numbers.Real = 0.0,
        ub: numbers.Real = mip.INF,
        obj: numbers.Real = 0.0,
        var_type: str = mip.CONTINUOUS,
        column: "mip.Column" = None,
    ) -> "mip.Var":
        solver = self.__model.solver
        if not name:
            name = "var({})".format(len(self))
        if var_type == mip.BINARY:
            lb = 0.0
            ub = 1.0
        new_var = mip.Var(self.__model, solver.num_cols())
        solver.add_var(obj, lb, ub, var_type, column, name)
        return new_var

    def __getitem__(self: "VVarList", key):
        if isinstance(key, str):
            return self.__model.var_by_name(key)
        if isinstance(key, slice):
            return VVarList(self.__model, key.start, key.end)
        if isinstance(key, int):
            if key < 0:
                key = self.__end - key
            if key >= self.__end:
                raise IndexError

            return mip.Var(self.__model, key + self.__start)

        raise TypeError("Unknown type {}".format(type(key)))

    def __len__(self: "VVarList") -> int:
        return self.__model.solver.num_cols()


class ConstrList(Sequence):
    """ List of problem constraints"""

    def __init__(self: "ConstrList", model: "mip.Model"):
        self.__model = model
        self.__constrs = []  # type: List["mip.Constr"]

    def __getitem__(self: "ConstrList", key):
        if isinstance(key, str):
            return self.__model.constr_by_name(key)
        return self.__constrs[key]

    def add(self, lin_expr: "mip.LinExpr", name: str = "") -> "mip.Constr":
        if not name:
            name = "constr({})".format(len(self.__constrs))
        new_constr = mip.Constr(self.__model, len(self.__constrs))
        self.__model.solver.add_constr(lin_expr, name)
        self.__constrs.append(new_constr)
        return new_constr

    def __len__(self) -> int:
        return len(self.__constrs)

    def remove(self: "ConstrList", constrs: List["mip.Constr"]):
        iv = [1 for i in range(len(self.__constrs))]
        clist = [c.idx for c in constrs]
        clist.sort()
        for i in clist:
            iv[i] = 0
        self.__model.solver.remove_constrs(clist)
        i = 0
        for c in self.__constrs:
            if iv[c.idx] == 0:
                c.idx = -1
            else:
                c.idx = i
                i += 1
        self.__constrs = [c for c in self.__constrs if c.idx != -1]

    def update_constrs(self: "ConstrList", n_constrs: int):
        self.__constrs = [mip.Constr(self.__model, i) for i in range(n_constrs)]


# same as previous class, but does not stores
# anything and does not allows modification,
# used in callbacks
class VConstrList(Sequence):
    def __init__(self: "VConstrList", model: "mip.Model"):
        self.__model = model

    def __getitem__(self: "VConstrList", key):
        if isinstance(key, str):
            return self.__model.constr_by_name(key)
        elif isinstance(key, int):
            return mip.Constr(self.__model, key)
        elif isinstance(key, slice):
            return self[key]

        raise TypeError("Use int, string or slice as key")

    def __len__(self) -> int:
        return self.__model.solver.num_rows()


class EmptyVarSol(Sequence):
    """A list that always returns None when acessed, just to be used
    when no solution is available."""

    def __init__(self: "EmptyVarSol", model: "mip.Model"):
        self.__model = model

    def __len__(self) -> int:
        return self.__model.solver.num_cols()

    def __getitem__(self: "EmptyVarSol", key):
        return None


class EmptyRowSol(Sequence):
    """A list that always returns None when acessed, just to be used
    when no solution is available."""

    def __init__(self: "EmptyRowSol", model: "mip.Model"):
        self.__model = model

    def __len__(self) -> int:
        return self.__model.solver.num_rows()

    def __getitem__(self: "EmptyRowSol", key):
        return None
