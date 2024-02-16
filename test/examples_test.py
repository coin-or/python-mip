"""Tests all examples in the ../examples folder
with all the solvers available"""
from glob import glob
from os.path import join
from os import environ
from itertools import product
import types
import importlib.machinery
import pytest

import mip.gurobi
import mip.highs
from mip import CBC, GUROBI, HIGHS

EXAMPLES = glob(join("..", "examples", "*.py")) + glob(join(".", "examples", "*.py"))

SOLVERS = [CBC]
if mip.gurobi.has_gurobi:
    SOLVERS += [GUROBI]
if mip.highs.has_highs:
    SOLVERS += [HIGHS]


@pytest.mark.parametrize("solver, example", product(SOLVERS, EXAMPLES))
def test_examples(solver, example):
    """Executes a given example with using solver 'solver'"""
    environ["SOLVER_NAME"] = solver
    loader = importlib.machinery.SourceFileLoader("example", example)
    mod = types.ModuleType(loader.name)
    try:
        loader.exec_module(mod)
    except NotImplementedError as e:
        print("Skipping test for example '{}': {}".format(example, e))
