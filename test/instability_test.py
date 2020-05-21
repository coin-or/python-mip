import pytest
from mip import Model, xsum, OptimizationStatus, MAXIMIZE, BINARY, INTEGER
from mip import ConstrsGenerator, CutPool, maximize, CBC, GUROBI, Column
from mip.ndarray import LinExprTensor
from os import environ
import time


def test_windows_instability():
    model = Model()
    model.read("./data/windows-instability.mps.mps.gz")
    result = model.optimize(max_seconds=300)

    assert result in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE)


if __name__ == "__main__":
    test_windows_instability()
