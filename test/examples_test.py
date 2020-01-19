"""Tests all examples in the ../examples folder
with all the solvers available"""
from glob import glob
from os.path import join
from os import environ
from itertools import product
import imp
import pytest


EXAMPLES = glob(join("..", "examples", "*.py")) + glob(join(".", "examples", "*.py"))

SOLVERS = ["cbc"]
if "GUROBI_HOME" in environ:
    SOLVERS += ["gurobi"]


@pytest.mark.parametrize("solver, example", product(SOLVERS, EXAMPLES))
def test_examples(solver, example):
    environ["SOLVER_NAME"] = solver
    m = imp.load_source("a", example)
