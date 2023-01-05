"Python-MIP interface to the HiGHS solver."

import glob
import logging
import os
import os.path
import sys

import cffi

import mip

logger = logging.getLogger(__name__)

# try loading the solver library
ffi = cffi.FFI()
try:
    # first try user-defined path, if given
    ENV_KEY = "PMIP_HIGHS_LIBRARY"
    if ENV_KEY in os.environ:
        libfile = os.environ[ENV_KEY]
        logger.debug("Choosing HiGHS library {libfile} via {ENV_KEY}.")
    else:
        # try library shipped with highspy packaged
        import highspy

        # HACK: find dynamic library in sibling folder
        pkg_path = os.path.dirname(highspy.__file__)
        libs_path = f"{pkg_path}.libs"
        # need library matching operating system
        if "linux" in sys.platform.lower():
            pattern = "libhighs-*.so.*"
        else:
            raise NotImplementedError(f"{sys.platform} not supported!")
        # there should only be one match
        [libfile] = glob.glob(os.path.join(libs_path, pattern))
        logger.debug("Choosing HiGHS library {libfile} via highspy package.")

    highslib = ffi.dlopen(libfile)
    has_highs = True
except Exception as e:
    logger.error(f"An error occurred while loading the HiGHS library:\n{e}")
    has_highs = False


class SolverHighs(mip.Solver):
    pass  # TODO
