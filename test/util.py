from functools import lru_cache, wraps

import pytest

import mip
import mip.gurobi


def skip_on(exception):
    """
    Skips the test in case the given exception is raised.
    :param exception: exception to consider
    :return: decorator function
    """

    def decorator_func(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exception as e:
                pytest.skip(str(e))

        return wrapper

    return decorator_func


@lru_cache(maxsize=1)
def has_gurobi_license():
    if not mip.gurobi.has_gurobi:
        return False

    try:
        mip.Model(solver_name=mip.GUROBI)
    except mip.InterfacingError:
        return False

    return True
