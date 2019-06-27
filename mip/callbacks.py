"""Classes used in solver callbacks, for a bi-directional communication
with the solver engine"""
from collections import defaultdict
from typing import List, Tuple
from mip.model import Model, LinExpr


class CallbackModel(Model):
    def __init__(self, model, where=None):
        self.__model = model
        self.__where = where  # used to determine the type of callback


class CutsGenerator:
    """abstract class for implementing cut generators"""

    def __init__(self):
        pass

    def generate_cuts(self, model: Model):
        """Method called by the solver engine to generate cuts

           After analyzing the contents of the fractional solution in model
           variables :meth:`~mip.model.Model.vars`, whose solution values can
           be queried with the in :meth:`~mip.model.Var.x` method, one or more
           cuts may be generated and added to the solver engine cut pool with
           the :meth:`~mip.model.Model.add_cut` method.

        Args:

            model(Model): model for which cuts may be generated. Please note
                that this model may have fewer variables than the original
                model due to pre-processing. If you want to generate cuts
                in terms of the original variables, one alternative is to
                query variables by their names, checking which ones remain
                in this pre-processed problem. In this procedure you can
                query model properties and add cuts
                (:meth:`~mip.model.Model.add_cut`), but you cannot perform
                other model modifications, such as add columns.
        """
        raise NotImplementedError()


class BranchSelector:
    def __init__(self, model: "Model"):
        self.model = model

    def select_branch(self, rsol: List[Tuple["Var", float]]) \
            -> Tuple["Var", int]:
        raise NotImplementedError()


class CutPool:
    def __init__(self):
        """Stores a list list of different cuts, repeated cuts are discarded.
        """
        self.__cuts = []

        # positions for each hash code to speedup
        # the search of repeated cuts
        self.__pos = defaultdict(list)

    def add(self, cut: LinExpr) -> bool:
        """tries to add a cut to the pool, returns true if this is a new cut,
        false if it is a repeated one

        Args:
            cut(LinExpr): a constraint
        """
        hcode = hash(cut)
        bucket = self.__pos[hcode]
        for p in bucket:
            if self.__cuts[p].equals(cut):
                return False

        self.__pos[hcode].append(len(self.__cuts))
        self.__cuts.append(cut)

        return True

    @property
    def cuts(self) -> List["LinExpr"]:
        return self.__cuts


class IncumbentUpdater:
    """To receive notifications whenever a new integer feasible solution is
    found. Optionally a new improved solution can be generated (using some
    local search heuristic) and returned to the MIP solver.
    """

    def __init__(self, model: "Model"):
        self.model = model

    def update_incumbent(self, objective_value: float, best_bound: float,
                         solution: List[Tuple["Var", float]]) \
            -> List[Tuple["Var", float]]:
        """method that is called when a new integer feasible solution is found

        Args:
            objective_value(float): cost of the new solution found
            best_bound(float): current lower bound for the optimal solution
            cost solution(List[Tuple[Var,float]]): non-zero variables
            in the solution
        """
        raise NotImplementedError()


class LazyConstrsGenerator:
    def __init(self, model: "Model"):
        self.model = model

    def generate_lazy_constrs(self, solution: List[Tuple["Var", float]]
                              ) -> List["LinExpr"]:
        raise NotImplementedError()
