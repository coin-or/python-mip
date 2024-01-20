import pytest

from functools import wraps


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
