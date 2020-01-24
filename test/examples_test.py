"""Tests all examples in the ../examples folder
with all the solvers available"""
from glob import glob
from os.path import join
from os import environ
from itertools import product
import types
import importlib.machinery
import pytest


EXAMPLES = glob(join("..", "examples", "*.py")) + glob(join(".", "examples", "*.py"))

SOLVERS = ["cbc"]
if "GUROBI_HOME" in environ:
    SOLVERS += ["gurobi"]


@pytest.mark.parametrize("solver, example", product(SOLVERS, EXAMPLES))
def test_examples(solver, example):
    """Executes a given example with using solver 'solver'"""
    environ["SOLVER_NAME"] = solver
    loader = importlib.machinery.SourceFileLoader("example", example)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
