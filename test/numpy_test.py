from itertools import product
import pytest
import numpy as np
from mip import Model, xsum, OptimizationStatus, MAXIMIZE, BINARY, INTEGER
from mip import ConstrsGenerator, CutPool, maximize, CBC, GUROBI, Column
from mip.ndarray import LinExprTensor
from os import environ
import time
import sys


def test_numpy():
    model = Model()
    N = 1000

    start = time.time()
    x = model.add_var_tensor(shape=(N, N), name="x")

    # inefficient way to compute trace, so we can test optimizations
    # equivalent to model += np.trace(x)
    model += np.ones((N,)) @ (x * np.eye(N)) @ np.ones((N,))

    # constraints
    model += np.vectorize(lambda x_i_j: x_i_j >= 1)(x)

    stop = time.time()
    print("model built in: %.1f seconds" % (stop - start))

    model.write("numpy_tensors.lp")
    result = model.optimize()

    assert result == OptimizationStatus.OPTIMAL


def test_LinExprTensor():
    model = Model()
    x = model.add_var_tensor(shape=(3,), name="x")
    print(x)
    assert x.shape == (3,)
    assert isinstance(x, LinExprTensor)

    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = A @ x
    print(y)
    assert y.shape == (3,)
    assert isinstance(y, LinExprTensor)

    constr = y <= 10
    print(constr)
    assert constr.shape == (3,)
    assert isinstance(x, LinExprTensor)

    constr = y >= 10
    print(constr)
    assert constr.shape == (3,)
    assert isinstance(x, LinExprTensor)

    constr = y == 10
    print(constr)
    assert constr.shape == (3,)
    assert isinstance(x, LinExprTensor)


if __name__ == "__main__":
    test_numpy()
