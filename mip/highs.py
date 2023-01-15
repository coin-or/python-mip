"Python-MIP interface to the HiGHS solver."

import glob
import numbers
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

        pkg_path = os.path.dirname(highspy.__file__)

        # need library matching operating system
        if "linux" in sys.platform.lower():
            pattern = "highs_bindings.*.so"
        else:
            raise NotImplementedError(f"{sys.platform} not supported!")

        # there should only be one match
        [libfile] = glob.glob(os.path.join(pkg_path, pattern))
        logger.debug("Choosing HiGHS library {libfile} via highspy package.")

    highslib = ffi.dlopen(libfile)
    has_highs = True
except Exception as e:
    logger.error(f"An error occurred while loading the HiGHS library:\n{e}")
    has_highs = False

HEADER = """
typedef int HighsInt;

void* Highs_create(void);
void Highs_destroy(void* highs);
HighsInt Highs_readModel(void* highs, const char* filename);
HighsInt Highs_writeModel(void* highs, const char* filename);
HighsInt Highs_run(void* highs);
HighsInt Highs_getModelStatus(const void* highs);
double Highs_getObjectiveValue(const void* highs);
HighsInt Highs_addVar(void* highs, const double lower, const double upper);
HighsInt Highs_addRow(
    void* highs, const double lower, const double upper, const HighsInt num_new_nz,
    const HighsInt* index, const double* value
);
HighsInt Highs_changeColIntegrality(
    void* highs, const HighsInt col, const HighsInt integrality
);
HighsInt Highs_changeColCost(void* highs, const HighsInt col, const double cost);
HighsInt Highs_changeColBounds(
    void* highs, const HighsInt col, const double lower, const double upper
);
HighsInt Highs_getRowsByRange(
    const void* highs, const HighsInt from_row, const HighsInt to_row,
    HighsInt* num_row, double* lower, double* upper, HighsInt* num_nz,
    HighsInt* matrix_start, HighsInt* matrix_index, double* matrix_value
);
"""

if has_highs:
    ffi.cdef(HEADER)


class SolverHighs(mip.Solver):
    def __init__(self, model: mip.Model, name: str, sense: str):
        super().__init__(model, name, sense)

        # Store reference to library so that it's not garbage-collected (when we
        # just use highslib in __del__, it had already become None)?!
        self._lib = highslib

        self._model = highslib.Highs_create()

    def __del__(self):
        self._lib.Highs_destroy(self._model)

    def add_var(
        self,
        name: str = "",
        obj: numbers.Real = 0,
        lb: numbers.Real = 0,
        ub: numbers.Real = mip.INF,
        var_type: str = mip.CONTINUOUS,
        column: "Column" = None,
    ):
        pass


# create solver for testing
if has_highs:
    print("have highs")
    model = None
    solver = SolverHighs(None, "foo_name", mip.MINIMIZE)
else:
    print("don't have highs")
