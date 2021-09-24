"""This code shows five different ways for creating a particular set of linear
expressions. Note that some of these 'ways' result in very poor performance,
but should work and therefore are included for testing purposes.

Note: the code is based on issue #163
Contributors: @jachymb and @christian2022
"""
from collections import OrderedDict
import mip, random, time


def using_addvar(n):
    model = mip.Model(sense=mip.MAXIMIZE)
    model.verbose = False

    start = time.time()

    obj = mip.LinExpr()

    for i in range(n):
        var = model.add_var(lb=0.5, ub=1.0)
        a = random.random() - 0.5  # Just a dummy value for illustration
        obj.add_var(var, a)
        model += obj <= 1.0
    model.objective = obj

    construction_time = time.time() - start

    start = time.time()
    model.optimize()
    solution_time = time.time() - start

    return model.objective_value, construction_time, solution_time


def using_inplace_op(n):
    model = mip.Model(sense=mip.MAXIMIZE)
    model.verbose = False

    start = time.time()

    obj = mip.LinExpr()

    for i in range(n):
        var = model.add_var(lb=0.5, ub=1.0)
        a = random.random() - 0.5  # Just a dummy value for illustration
        obj += var * a
        model += obj <= 1.0
    model.objective = obj

    construction_time = time.time() - start

    start = time.time()
    model.optimize()
    solution_time = time.time() - start

    return model.objective_value, construction_time, solution_time


def using_lists(n):
    model = mip.Model(sense=mip.MAXIMIZE)
    model.verbose = False

    start = time.time()

    variables = []
    coeffs = []

    for i in range(n):
        variables.append(model.add_var(lb=0.5, ub=1.0))
        coeffs.append(random.random() - 0.5)  # Just a dummy value for illustration
        model += mip.LinExpr(variables=variables, coeffs=coeffs) <= 1.0
    model.objective = mip.LinExpr(variables=variables, coeffs=coeffs)

    construction_time = time.time() - start

    start = time.time()
    model.optimize()
    solution_time = time.time() - start

    return model.objective_value, construction_time, solution_time


def using_dict(n):
    model = mip.Model(sense=mip.MAXIMIZE)
    model.verbose = False

    start = time.time()

    expr = {}

    for i in range(n):
        var = model.add_var(lb=0.5, ub=1.0)
        a = random.random() - 0.5  # Just a dummy value for illustration
        expr[var] = a
        model += mip.LinExpr(expr=expr) <= 1.0
    model.objective = mip.LinExpr(expr=expr)

    construction_time = time.time() - start

    start = time.time()
    model.optimize()
    solution_time = time.time() - start

    return model.objective_value, construction_time, solution_time


def recreating_linexpr(n):
    model = mip.Model(sense=mip.MAXIMIZE)
    model.verbose = False

    start = time.time()

    obj = []

    for i in range(n):
        var = model.add_var(lb=0.5, ub=1.0)
        a = random.random() - 0.5  # Just a dummy value for illustration
        obj.append(var * a)
        model += mip.xsum(obj) <= 1.0
    model.objective = mip.xsum(obj)

    construction_time = time.time() - start

    start = time.time()
    model.optimize()
    solution_time = time.time() - start

    return model.objective_value, construction_time, solution_time


sizes = [100, 500, 1000, 2000]
results = [0.897191700, 0.504060660, 0.990206267, 0.715224329]
functions = [
    recreating_linexpr,
    using_addvar,
    using_inplace_op,
    using_lists,
    using_dict,
]

runtimes = OrderedDict()

for k, result in zip(sizes, results):
    runtimes[k] = OrderedDict()

    for function in functions:
        random.seed(0)  # resetting seed to make result predictable
        cost, construction_time, solution_time = function(k)
        if result is not None:
            assert result - 1e-9 <= cost <= result + 1e-9

        runtimes[k][function.__name__] = construction_time

# printing report in the end
for k in runtimes:
    print("n =", k)
    for function, runtime in runtimes[k].items():
        print("   ", function, ":", runtime)
