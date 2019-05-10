from collections import defaultdict
from mip.model import *
from typing import List, Tuple


class BranchSelector:
    def __init__(self, model: "Model"):
        self.model = model

    def select_branch(self,
                      rsol: List[Tuple["Var", float]]) -> Tuple["Var", int]:
        raise NotImplementedError()


class CutsGenerator:
    """abstract class for implementing cut generators"""
    def __init__(self, model: "Model"):
        self.model = model

    def generate_cuts(self,
                      rsol: List[Tuple["Var", float]]) -> List["LinExpr"]:
        """Method called by the solve engine to generate cuts

           After analyzing the contents of the fractional solution in
           :code:`relax_solution`, one or mode cuts
           (:class:`~mip.model.LinExpr`) may be generated and returned.
           These cuts are added to the relaxed model.

        Args:
            rsol(List[Tuple[Var, float]]): a list of tuples (variable,value)
            indicating the values of variables in the current fractional
            solution. Variables at zero are not included.

        Note: take care not to query the value of the fractional solution in
        the cut generation method using the :code:`x` methods from original
        references to problem variables, use the contents of
        :code:`relax_solution` instead.
        """
        raise NotImplementedError()


class CutPool:
    def __init__(self: "CutPool"):
        """Stores a list list of different cuts, repeated cuts are discarded.
        """
        self.__cuts = []

        # positions for each hash code to speedup
        # the search of repeated cuts
        self.__pos = defaultdict(list)

    def add(self: "CutPool", cut: "LinExpr") -> bool:
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
    def cuts(self: "CutPool") -> List["LinExpr"]:
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

    def generate_lazy_constrs(self, solution: List[Tuple["Var", float]]) -> List["LinExpr"]:
        raise NotImplementedError()
