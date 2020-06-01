import sys


def pytest_configure(config):
    import sys
    sys._called_from_test = True


def pytest_unconfigure(config):
    if hasattr(sys, '_called_from_test'):
        del sys._called_from_test
