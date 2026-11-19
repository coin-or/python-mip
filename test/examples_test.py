"""Tests all examples in the ../examples folder
with all the solvers available"""
from glob import glob
from os.path import basename, join
from itertools import product
import importlib.util
import pytest

import mip.gurobi
import mip.highs
import mip.scip
from mip import CBC, GUROBI, HIGHS, SCIP
from util import skip_on, has_gurobi_license

EXAMPLES = glob(join("..", "examples", "*.py")) + glob(join(".", "examples", "*.py"))

SOLVERS = [CBC]
if has_gurobi_license():
    SOLVERS += [GUROBI]
if mip.highs.has_highs:
    SOLVERS += [HIGHS]
if mip.scip.has_scip:
    SOLVERS += [SCIP]


@skip_on(NotImplementedError)
@pytest.mark.parametrize("solver, example", product(SOLVERS, EXAMPLES))
def test_examples(solver, example, monkeypatch):
    """Executes a given example with using solver 'solver'"""
    if basename(example) in {"conflict_finder.py", "numpy_tensor.py"}:
        pytest.importorskip("numpy")

    monkeypatch.setenv("SOLVER_NAME", solver)
    spec = importlib.util.spec_from_file_location("example", example)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load example {example}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
