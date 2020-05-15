from itertools import product
import pytest
import numpy as np
from mip import Model, xsum, OptimizationStatus, MAXIMIZE, BINARY, INTEGER
from mip import ConstrsGenerator, CutPool, maximize, CBC, GUROBI, Column
from os import environ
import time

def test_numpy():
    model = Model()
    N = 1000

    start = time.time()
    x = model.add_var_tensor(shape=(N, N), name='x')

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

if __name__ == "__main__":
    test_numpy()