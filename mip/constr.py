from collections.abc import Sequence
from typing import List


class Constr:
    """ A row (constraint) in the constraint matrix.

        A constraint is a specific :class:`~mip.model.LinExpr` that includes a
        sense (<, > or == or less-or-equal, greater-or-equal and equal,
        respectively) and a right-hand-side constant value. Constraints can be
        added to the model using the overloaded operator :code:`+=` or using
        the method :meth:`~mip.model.Model.add_constr` of the
        :class:`~mip.model.Model` class:

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
            res = self.name + ':'
        else:
            res = 'constr({}): '.format(self.idx + 1)
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
        rhs = self.expr.const * -1.0
        if self.expr.sense == '=':
            res += ' = {}'.format(rhs)
        elif self.expr.sense == '<':
            res += ' <= {}'.format(rhs)
        elif self.expr.sense == '>':
            res += ' >= {}'.format(rhs)

        return res

    @property
    def slack(self) -> float:

        """Value of the slack in this constraint in the optimal
        solution. Available only if the formulation was solved.
        """

        return self.__model.solver.constr_get_slack(self)

    @property
    def pi(self) -> float:

        """Value for the dual variable of this constraint in the optimal
        solution of a linear programming :class:`~mip.model.Model`. Only
        available if a pure linear programming problem was solved (only
        continuous variables).
        """

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


class ConstrList(Sequence):
    """ List of problem constraints"""

    def __init__(self, model: "Model"):
        self.__model = model
        self.__constrs = []

    def __getitem__(self, key):
        if (isinstance(key, str)):
            return self.__model.constr_by_name(key)
        return self.__constrs[key]

    def add(self,
            lin_expr: "LinExpr",
            name: str = '') -> Constr:
        if not name:
            name = 'constr({})'.format(len(self.__constrs))
        new_constr = Constr(self.__model, len(self.__constrs))
        self.__model.solver.add_constr(lin_expr, name)
        self.__constrs.append(new_constr)
        return new_constr

    def __len__(self) -> int:
        return len(self.__constrs)

    def remove(self, constrs: List[Constr]):
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
        self.__constrs = [c for c in
                          self.__constrs
                          if c.idx != -1]

    def update_constrs(self, n_constrs: int):
        self.__constrs = [Constr(self.__model, i) for i in range(n_constrs)]


# same as previous class, but does not stores
# anything and does not allows modification,
# used in callbacks
class VConstrList(Sequence):

    def __init__(self, model: "Model"):
        self.__model = model

    def __getitem__(self, key):
        if (isinstance(key, str)):
            return self.__model.constr_by_name(key)
        elif (isinstance(key, int)):
            return Constr(self.__model, key)
        elif (isinstance(key, slice)):
            return self[key]

        raise Exception('Use int or string as key')

    def __len__(self) -> int:
        return self.__model.solver.num_rows()
