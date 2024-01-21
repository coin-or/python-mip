"Python-MIP interface to the HiGHS solver."

import glob
import numbers
import logging
import os.path
import sys
from typing import List, Optional, Tuple, Union

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
        platform = sys.platform.lower()
        if "linux" in platform:
            pattern = "highs_bindings.*.so"
        elif platform.startswith("win"):
            pattern = "highs_bindings.*.pyd"
        elif any(platform.startswith(p) for p in ("darwin", "macos")):
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

if has_highs:
    ffi.cdef(
        """
        /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
        /*                                                                       */
        /*    This file is part of the HiGHS linear optimization suite           */
        /*                                                                       */
        /*    Written and engineered 2008-2024 by Julian Hall, Ivet Galabova,    */
        /*    Leona Gottwald and Michael Feldmeier                               */
        /*                                                                       */
        /*    Available as open-source under the MIT License                     */
        /*                                                                       */
        /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
        typedef int HighsInt;
        
        typedef struct {
          int log_type;  // cast of HighsLogType
          double running_time;
          HighsInt simplex_iteration_count;
          HighsInt ipm_iteration_count;
          double objective_function_value;
          int64_t mip_node_count;
          double mip_primal_bound;
          double mip_dual_bound;
          double mip_gap;
          double* mip_solution;
        } HighsCallbackDataOut;
        
        typedef struct {
          int user_interrupt;
        } HighsCallbackDataIn;
        
        typedef void (*HighsCCallbackType)(int, const char*,
                                           const HighsCallbackDataOut*,
                                           HighsCallbackDataIn*, void*);

        const HighsInt kHighsMaximumStringLength = 512;
        
        const HighsInt kHighsStatusError = -1;
        const HighsInt kHighsStatusOk = 0;
        const HighsInt kHighsStatusWarning = 1;
        
        const HighsInt kHighsVarTypeContinuous = 0;
        const HighsInt kHighsVarTypeInteger = 1;
        const HighsInt kHighsVarTypeSemiContinuous = 2;
        const HighsInt kHighsVarTypeSemiInteger = 3;
        const HighsInt kHighsVarTypeImplicitInteger = 4;
        
        const HighsInt kHighsOptionTypeBool = 0;
        const HighsInt kHighsOptionTypeInt = 1;
        const HighsInt kHighsOptionTypeDouble = 2;
        const HighsInt kHighsOptionTypeString = 3;
        
        const HighsInt kHighsInfoTypeInt64 = -1;
        const HighsInt kHighsInfoTypeInt = 1;
        const HighsInt kHighsInfoTypeDouble = 2;
        
        const HighsInt kHighsObjSenseMinimize = 1;
        const HighsInt kHighsObjSenseMaximize = -1;
        
        const HighsInt kHighsMatrixFormatColwise = 1;
        const HighsInt kHighsMatrixFormatRowwise = 2;
        
        const HighsInt kHighsHessianFormatTriangular = 1;
        const HighsInt kHighsHessianFormatSquare = 2;
        
        const HighsInt kHighsSolutionStatusNone = 0;
        const HighsInt kHighsSolutionStatusInfeasible = 1;
        const HighsInt kHighsSolutionStatusFeasible = 2;
        
        const HighsInt kHighsBasisValidityInvalid = 0;
        const HighsInt kHighsBasisValidityValid = 1;
        
        const HighsInt kHighsPresolveStatusNotPresolved = -1;
        const HighsInt kHighsPresolveStatusNotReduced = 0;
        const HighsInt kHighsPresolveStatusInfeasible = 1;
        const HighsInt kHighsPresolveStatusUnboundedOrInfeasible = 2;
        const HighsInt kHighsPresolveStatusReduced = 3;
        const HighsInt kHighsPresolveStatusReducedToEmpty = 4;
        const HighsInt kHighsPresolveStatusTimeout = 5;
        const HighsInt kHighsPresolveStatusNullError = 6;
        const HighsInt kHighsPresolveStatusOptionsError = 7;
        
        const HighsInt kHighsModelStatusNotset = 0;
        const HighsInt kHighsModelStatusLoadError = 1;
        const HighsInt kHighsModelStatusModelError = 2;
        const HighsInt kHighsModelStatusPresolveError = 3;
        const HighsInt kHighsModelStatusSolveError = 4;
        const HighsInt kHighsModelStatusPostsolveError = 5;
        const HighsInt kHighsModelStatusModelEmpty = 6;
        const HighsInt kHighsModelStatusOptimal = 7;
        const HighsInt kHighsModelStatusInfeasible = 8;
        const HighsInt kHighsModelStatusUnboundedOrInfeasible = 9;
        const HighsInt kHighsModelStatusUnbounded = 10;
        const HighsInt kHighsModelStatusObjectiveBound = 11;
        const HighsInt kHighsModelStatusObjectiveTarget = 12;
        const HighsInt kHighsModelStatusTimeLimit = 13;
        const HighsInt kHighsModelStatusIterationLimit = 14;
        const HighsInt kHighsModelStatusUnknown = 15;
        const HighsInt kHighsModelStatusSolutionLimit = 16;
        const HighsInt kHighsModelStatusInterrupt = 17;
        
        const HighsInt kHighsBasisStatusLower = 0;
        const HighsInt kHighsBasisStatusBasic = 1;
        const HighsInt kHighsBasisStatusUpper = 2;
        const HighsInt kHighsBasisStatusZero = 3;
        const HighsInt kHighsBasisStatusNonbasic = 4;
        
        const HighsInt kHighsCallbackLogging = 0;
        const HighsInt kHighsCallbackSimplexInterrupt = 1;
        const HighsInt kHighsCallbackIpmInterrupt = 2;
        const HighsInt kHighsCallbackMipSolution = 3;
        const HighsInt kHighsCallbackMipImprovingSolution = 4;
        const HighsInt kHighsCallbackMipLogging = 5;
        const HighsInt kHighsCallbackMipInterrupt = 6;
        
        HighsInt Highs_lpCall(const HighsInt num_col, const HighsInt num_row,
                              const HighsInt num_nz, const HighsInt a_format,
                              const HighsInt sense, const double offset,
                              const double* col_cost, const double* col_lower,
                              const double* col_upper, const double* row_lower,
                              const double* row_upper, const HighsInt* a_start,
                              const HighsInt* a_index, const double* a_value,
                              double* col_value, double* col_dual, double* row_value,
                              double* row_dual, HighsInt* col_basis_status,
                              HighsInt* row_basis_status, HighsInt* model_status);
        
        HighsInt Highs_mipCall(const HighsInt num_col, const HighsInt num_row,
                               const HighsInt num_nz, const HighsInt a_format,
                               const HighsInt sense, const double offset,
                               const double* col_cost, const double* col_lower,
                               const double* col_upper, const double* row_lower,
                               const double* row_upper, const HighsInt* a_start,
                               const HighsInt* a_index, const double* a_value,
                               const HighsInt* integrality, double* col_value,
                               double* row_value, HighsInt* model_status);
        
        HighsInt Highs_qpCall(
            const HighsInt num_col, const HighsInt num_row, const HighsInt num_nz,
            const HighsInt q_num_nz, const HighsInt a_format, const HighsInt q_format,
            const HighsInt sense, const double offset, const double* col_cost,
            const double* col_lower, const double* col_upper, const double* row_lower,
            const double* row_upper, const HighsInt* a_start, const HighsInt* a_index,
            const double* a_value, const HighsInt* q_start, const HighsInt* q_index,
            const double* q_value, double* col_value, double* col_dual,
            double* row_value, double* row_dual, HighsInt* col_basis_status,
            HighsInt* row_basis_status, HighsInt* model_status);
        
        void* Highs_create(void);
        
        void Highs_destroy(void* highs);
        
        const char* Highs_version(void);
        
        HighsInt Highs_versionMajor(void);
        
        HighsInt Highs_versionMinor(void);
        
        HighsInt Highs_versionPatch(void);
        
        const char* Highs_githash(void);
        
        const char* Highs_compilationDate(void);
        
        HighsInt Highs_readModel(void* highs, const char* filename);
        
        HighsInt Highs_writeModel(void* highs, const char* filename);
        
        HighsInt Highs_clear(void* highs);
        
        HighsInt Highs_clearModel(void* highs);
        
        HighsInt Highs_clearSolver(void* highs);
        
        HighsInt Highs_run(void* highs);
        
        HighsInt Highs_writeSolution(const void* highs, const char* filename);
        
        HighsInt Highs_writeSolutionPretty(const void* highs, const char* filename);
        
        HighsInt Highs_passLp(void* highs, const HighsInt num_col,
                              const HighsInt num_row, const HighsInt num_nz,
                              const HighsInt a_format, const HighsInt sense,
                              const double offset, const double* col_cost,
                              const double* col_lower, const double* col_upper,
                              const double* row_lower, const double* row_upper,
                              const HighsInt* a_start, const HighsInt* a_index,
                              const double* a_value);
        
        HighsInt Highs_passMip(void* highs, const HighsInt num_col,
                               const HighsInt num_row, const HighsInt num_nz,
                               const HighsInt a_format, const HighsInt sense,
                               const double offset, const double* col_cost,
                               const double* col_lower, const double* col_upper,
                               const double* row_lower, const double* row_upper,
                               const HighsInt* a_start, const HighsInt* a_index,
                               const double* a_value, const HighsInt* integrality);
        
        HighsInt Highs_passModel(void* highs, const HighsInt num_col,
                                 const HighsInt num_row, const HighsInt num_nz,
                                 const HighsInt q_num_nz, const HighsInt a_format,
                                 const HighsInt q_format, const HighsInt sense,
                                 const double offset, const double* col_cost,
                                 const double* col_lower, const double* col_upper,
                                 const double* row_lower, const double* row_upper,
                                 const HighsInt* a_start, const HighsInt* a_index,
                                 const double* a_value, const HighsInt* q_start,
                                 const HighsInt* q_index, const double* q_value,
                                 const HighsInt* integrality);
        
        HighsInt Highs_passHessian(void* highs, const HighsInt dim,
                                   const HighsInt num_nz, const HighsInt format,
                                   const HighsInt* start, const HighsInt* index,
                                   const double* value);
        
        HighsInt Highs_passRowName(const void* highs, const HighsInt row,
                                   const char* name);
        
        HighsInt Highs_passColName(const void* highs, const HighsInt col,
                                   const char* name);
        
        HighsInt Highs_readOptions(const void* highs, const char* filename);
        
        HighsInt Highs_setBoolOptionValue(void* highs, const char* option,
                                          const HighsInt value);
        
        HighsInt Highs_setIntOptionValue(void* highs, const char* option,
                                         const HighsInt value);
        
        HighsInt Highs_setDoubleOptionValue(void* highs, const char* option,
                                            const double value);
        
        HighsInt Highs_setStringOptionValue(void* highs, const char* option,
                                            const char* value);
        
        HighsInt Highs_getBoolOptionValue(const void* highs, const char* option,
                                          HighsInt* value);
        
        HighsInt Highs_getIntOptionValue(const void* highs, const char* option,
                                         HighsInt* value);
        
        HighsInt Highs_getDoubleOptionValue(const void* highs, const char* option,
                                            double* value);
        
        HighsInt Highs_getStringOptionValue(const void* highs, const char* option,
                                            char* value);
        
        HighsInt Highs_getOptionType(const void* highs, const char* option,
                                     HighsInt* type);
        
        HighsInt Highs_resetOptions(void* highs);
        
        HighsInt Highs_writeOptions(const void* highs, const char* filename);
        
        HighsInt Highs_writeOptionsDeviations(const void* highs, const char* filename);
        
        HighsInt Highs_getNumOptions(const void* highs);
        
        HighsInt Highs_getOptionName(const void* highs, const HighsInt index,
                                     char** name);
        
        HighsInt Highs_getBoolOptionValues(const void* highs, const char* option,
                                           HighsInt* current_value,
                                           HighsInt* default_value);
        HighsInt Highs_getIntOptionValues(const void* highs, const char* option,
                                          HighsInt* current_value, HighsInt* min_value,
                                          HighsInt* max_value, HighsInt* default_value);
        
        HighsInt Highs_getDoubleOptionValues(const void* highs, const char* option,
                                             double* current_value, double* min_value,
                                             double* max_value, double* default_value);
        
        HighsInt Highs_getStringOptionValues(const void* highs, const char* option,
                                             char* current_value, char* default_value);
        
        HighsInt Highs_getIntInfoValue(const void* highs, const char* info,
                                       HighsInt* value);
        
        HighsInt Highs_getDoubleInfoValue(const void* highs, const char* info,
                                          double* value);
        
        HighsInt Highs_getInt64InfoValue(const void* highs, const char* info,
                                         int64_t* value);
        
        HighsInt Highs_getInfoType(const void* highs, const char* info, HighsInt* type);
        
        HighsInt Highs_getSolution(const void* highs, double* col_value,
                                   double* col_dual, double* row_value,
                                   double* row_dual);
        
        HighsInt Highs_getBasis(const void* highs, HighsInt* col_status,
                                HighsInt* row_status);
        
        HighsInt Highs_getModelStatus(const void* highs);
        
        HighsInt Highs_getDualRay(const void* highs, HighsInt* has_dual_ray,
                                  double* dual_ray_value);
        
        HighsInt Highs_getPrimalRay(const void* highs, HighsInt* has_primal_ray,
                                    double* primal_ray_value);
        
        double Highs_getObjectiveValue(const void* highs);
        
        HighsInt Highs_getBasicVariables(const void* highs, HighsInt* basic_variables);
        
        HighsInt Highs_getBasisInverseRow(const void* highs, const HighsInt row,
                                          double* row_vector, HighsInt* row_num_nz,
                                          HighsInt* row_index);
                                          
        HighsInt Highs_getBasisInverseCol(const void* highs, const HighsInt col,
                                          double* col_vector, HighsInt* col_num_nz,
                                          HighsInt* col_index);
        
        HighsInt Highs_getBasisSolve(const void* highs, const double* rhs,
                                     double* solution_vector, HighsInt* solution_num_nz,
                                     HighsInt* solution_index);
                                     
        HighsInt Highs_getBasisTransposeSolve(const void* highs, const double* rhs,
                                              double* solution_vector,
                                              HighsInt* solution_nz,
                                              HighsInt* solution_index);
                                              
        HighsInt Highs_getReducedRow(const void* highs, const HighsInt row,
                                     double* row_vector, HighsInt* row_num_nz,
                                     HighsInt* row_index);
                                     
        HighsInt Highs_getReducedColumn(const void* highs, const HighsInt col,
                                        double* col_vector, HighsInt* col_num_nz,
                                        HighsInt* col_index);
        
        HighsInt Highs_setBasis(void* highs, const HighsInt* col_status,
                                const HighsInt* row_status);
        
        HighsInt Highs_setLogicalBasis(void* highs);
        
        HighsInt Highs_setSolution(void* highs, const double* col_value,
                                   const double* row_value, const double* col_dual,
                                   const double* row_dual);
        
        HighsInt Highs_setCallback(void* highs, HighsCCallbackType user_callback,
                                   void* user_callback_data);
        
        HighsInt Highs_startCallback(void* highs, const int callback_type);
        
        HighsInt Highs_stopCallback(void* highs, const int callback_type);
        
        double Highs_getRunTime(const void* highs);
        
        HighsInt Highs_zeroAllClocks(const void* highs);
        
        HighsInt Highs_addCol(void* highs, const double cost, const double lower,
                              const double upper, const HighsInt num_new_nz,
                              const HighsInt* index, const double* value);
        
        HighsInt Highs_addCols(void* highs, const HighsInt num_new_col,
                               const double* costs, const double* lower,
                               const double* upper, const HighsInt num_new_nz,
                               const HighsInt* starts, const HighsInt* index,
                               const double* value);
        
        HighsInt Highs_addVar(void* highs, const double lower, const double upper);
        
        HighsInt Highs_addVars(void* highs, const HighsInt num_new_var,
                               const double* lower, const double* upper);
        
        HighsInt Highs_addRow(void* highs, const double lower, const double upper,
                              const HighsInt num_new_nz, const HighsInt* index,
                              const double* value);
        
        HighsInt Highs_addRows(void* highs, const HighsInt num_new_row,
                               const double* lower, const double* upper,
                               const HighsInt num_new_nz, const HighsInt* starts,
                               const HighsInt* index, const double* value);
        
        HighsInt Highs_changeObjectiveSense(void* highs, const HighsInt sense);
        
        HighsInt Highs_changeObjectiveOffset(void* highs, const double offset);
        
        HighsInt Highs_changeColIntegrality(void* highs, const HighsInt col,
                                            const HighsInt integrality);
        
        HighsInt Highs_changeColsIntegralityByRange(void* highs,
                                                    const HighsInt from_col,
                                                    const HighsInt to_col,
                                                    const HighsInt* integrality);
        
        HighsInt Highs_changeColsIntegralityBySet(void* highs,
                                                  const HighsInt num_set_entries,
                                                  const HighsInt* set,
                                                  const HighsInt* integrality);
        
        HighsInt Highs_changeColsIntegralityByMask(void* highs, const HighsInt* mask,
                                                   const HighsInt* integrality);
        
        HighsInt Highs_changeColCost(void* highs, const HighsInt col,
                                     const double cost);
        
        HighsInt Highs_changeColsCostByRange(void* highs, const HighsInt from_col,
                                             const HighsInt to_col, const double* cost);
        
        HighsInt Highs_changeColsCostBySet(void* highs, const HighsInt num_set_entries,
                                           const HighsInt* set, const double* cost);
        
        HighsInt Highs_changeColsCostByMask(void* highs, const HighsInt* mask,
                                            const double* cost);
        
        HighsInt Highs_changeColBounds(void* highs, const HighsInt col,
                                       const double lower, const double upper);
        
        HighsInt Highs_changeColsBoundsByRange(void* highs, const HighsInt from_col,
                                               const HighsInt to_col,
                                               const double* lower,
                                               const double* upper);
        
        HighsInt Highs_changeColsBoundsBySet(void* highs,
                                             const HighsInt num_set_entries,
                                             const HighsInt* set, const double* lower,
                                             const double* upper);
        
        HighsInt Highs_changeColsBoundsByMask(void* highs, const HighsInt* mask,
                                              const double* lower, const double* upper);
        
        HighsInt Highs_changeRowBounds(void* highs, const HighsInt row,
                                       const double lower, const double upper);
        
        HighsInt Highs_changeRowsBoundsBySet(void* highs,
                                             const HighsInt num_set_entries,
                                             const HighsInt* set, const double* lower,
                                             const double* upper);
        
        HighsInt Highs_changeRowsBoundsByMask(void* highs, const HighsInt* mask,
                                              const double* lower, const double* upper);
        
        HighsInt Highs_changeCoeff(void* highs, const HighsInt row, const HighsInt col,
                                   const double value);
        
        HighsInt Highs_getObjectiveSense(const void* highs, HighsInt* sense);
        
        HighsInt Highs_getObjectiveOffset(const void* highs, double* offset);
        
        HighsInt Highs_getColsByRange(const void* highs, const HighsInt from_col,
                                      const HighsInt to_col, HighsInt* num_col,
                                      double* costs, double* lower, double* upper,
                                      HighsInt* num_nz, HighsInt* matrix_start,
                                      HighsInt* matrix_index, double* matrix_value);
        
        HighsInt Highs_getColsBySet(const void* highs, const HighsInt num_set_entries,
                                    const HighsInt* set, HighsInt* num_col,
                                    double* costs, double* lower, double* upper,
                                    HighsInt* num_nz, HighsInt* matrix_start,
                                    HighsInt* matrix_index, double* matrix_value);
        
        HighsInt Highs_getColsByMask(const void* highs, const HighsInt* mask,
                                     HighsInt* num_col, double* costs, double* lower,
                                     double* upper, HighsInt* num_nz,
                                     HighsInt* matrix_start, HighsInt* matrix_index,
                                     double* matrix_value);
        
        HighsInt Highs_getRowsByRange(const void* highs, const HighsInt from_row,
                                      const HighsInt to_row, HighsInt* num_row,
                                      double* lower, double* upper, HighsInt* num_nz,
                                      HighsInt* matrix_start, HighsInt* matrix_index,
                                      double* matrix_value);
        
        HighsInt Highs_getRowsBySet(const void* highs, const HighsInt num_set_entries,
                                    const HighsInt* set, HighsInt* num_row,
                                    double* lower, double* upper, HighsInt* num_nz,
                                    HighsInt* matrix_start, HighsInt* matrix_index,
                                    double* matrix_value);
        
        HighsInt Highs_getRowsByMask(const void* highs, const HighsInt* mask,
                                     HighsInt* num_row, double* lower, double* upper,
                                     HighsInt* num_nz, HighsInt* matrix_start,
                                     HighsInt* matrix_index, double* matrix_value);
        HighsInt Highs_getRowName(const void* highs, const HighsInt row, char* name);
        
        HighsInt Highs_getRowByName(const void* highs, const char* name, HighsInt* row);
        
        HighsInt Highs_getColName(const void* highs, const HighsInt col, char* name);
        
        HighsInt Highs_getColByName(const void* highs, const char* name, HighsInt* col);
        
        HighsInt Highs_getColIntegrality(const void* highs, const HighsInt col,
                                         HighsInt* integrality);
        
        HighsInt Highs_deleteColsByRange(void* highs, const HighsInt from_col,
                                         const HighsInt to_col);
        
        HighsInt Highs_deleteColsBySet(void* highs, const HighsInt num_set_entries,
                                       const HighsInt* set);
        
        HighsInt Highs_deleteColsByMask(void* highs, HighsInt* mask);
        
        HighsInt Highs_deleteRowsByRange(void* highs, const int from_row,
                                         const HighsInt to_row);
        
        HighsInt Highs_deleteRowsBySet(void* highs, const HighsInt num_set_entries,
                                       const HighsInt* set);
        
        HighsInt Highs_deleteRowsByMask(void* highs, HighsInt* mask);
        
        HighsInt Highs_scaleCol(void* highs, const HighsInt col, const double scaleval);
        
        HighsInt Highs_scaleRow(void* highs, const HighsInt row, const double scaleval);
        
        double Highs_getInfinity(const void* highs);
        
        HighsInt Highs_getSizeofHighsInt(const void* highs);
        
        HighsInt Highs_getNumCol(const void* highs);
        
        HighsInt Highs_getNumRow(const void* highs);
        
        HighsInt Highs_getNumNz(const void* highs);
        
        HighsInt Highs_getHessianNumNz(const void* highs);
        
        HighsInt Highs_getModel(const void* highs, const HighsInt a_format,
                                const HighsInt q_format, HighsInt* num_col,
                                HighsInt* num_row, HighsInt* num_nz,
                                HighsInt* hessian_num_nz, HighsInt* sense,
                                double* offset, double* col_cost, double* col_lower,
                                double* col_upper, double* row_lower, double* row_upper,
                                HighsInt* a_start, HighsInt* a_index, double* a_value,
                                HighsInt* q_start, HighsInt* q_index, double* q_value,
                                HighsInt* integrality);
        
        HighsInt Highs_crossover(void* highs, const int num_col, const int num_row,
                                 const double* col_value, const double* col_dual,
                                 const double* row_dual);
        
        HighsInt Highs_getRanging(void* highs,
            double* col_cost_up_value, double* col_cost_up_objective,
            HighsInt* col_cost_up_in_var, HighsInt* col_cost_up_ou_var,
            double* col_cost_dn_value, double* col_cost_dn_objective,
            HighsInt* col_cost_dn_in_var, HighsInt* col_cost_dn_ou_var,
            double* col_bound_up_value, double* col_bound_up_objective,
            HighsInt* col_bound_up_in_var, HighsInt* col_bound_up_ou_var,
            double* col_bound_dn_value, double* col_bound_dn_objective,
            HighsInt* col_bound_dn_in_var, HighsInt* col_bound_dn_ou_var,
            double* row_bound_up_value, double* row_bound_up_objective,
            HighsInt* row_bound_up_in_var, HighsInt* row_bound_up_ou_var,
            double* row_bound_dn_value, double* row_bound_dn_objective,
            HighsInt* row_bound_dn_in_var, HighsInt* row_bound_dn_ou_var);
        
        void Highs_resetGlobalScheduler(const HighsInt blocking);
        
        // *********************
        // * Deprecated methods*
        // *********************

        const HighsInt HighsStatuskError = -1;
        const HighsInt HighsStatuskOk = 0;
        const HighsInt HighsStatuskWarning = 1;
        
        HighsInt Highs_call(const HighsInt num_col, const HighsInt num_row,
                            const HighsInt num_nz, const double* col_cost,
                            const double* col_lower, const double* col_upper,
                            const double* row_lower, const double* row_upper,
                            const HighsInt* a_start, const HighsInt* a_index,
                            const double* a_value, double* col_value, double* col_dual,
                            double* row_value, double* row_dual,
                            HighsInt* col_basis_status, HighsInt* row_basis_status,
                            HighsInt* model_status);
        
        HighsInt Highs_runQuiet(void* highs);
        
        HighsInt Highs_setHighsLogfile(void* highs, const void* logfile);
        
        HighsInt Highs_setHighsOutput(void* highs, const void* outputfile);
        
        HighsInt Highs_getIterationCount(const void* highs);
        
        HighsInt Highs_getSimplexIterationCount(const void* highs);
        
        HighsInt Highs_setHighsBoolOptionValue(void* highs, const char* option,
                                               const HighsInt value);
        
        HighsInt Highs_setHighsIntOptionValue(void* highs, const char* option,
                                              const HighsInt value);
        
        HighsInt Highs_setHighsDoubleOptionValue(void* highs, const char* option,
                                                 const double value);
        
        HighsInt Highs_setHighsStringOptionValue(void* highs, const char* option,
                                                 const char* value);
        
        HighsInt Highs_setHighsOptionValue(void* highs, const char* option,
                                           const char* value);
        
        HighsInt Highs_getHighsBoolOptionValue(const void* highs, const char* option,
                                               HighsInt* value);
        
        HighsInt Highs_getHighsIntOptionValue(const void* highs, const char* option,
                                              HighsInt* value);
        
        HighsInt Highs_getHighsDoubleOptionValue(const void* highs, const char* option,
                                                 double* value);
        
        HighsInt Highs_getHighsStringOptionValue(const void* highs, const char* option,
                                                 char* value);
        
        HighsInt Highs_getHighsOptionType(const void* highs, const char* option,
                                          HighsInt* type);
        
        HighsInt Highs_resetHighsOptions(void* highs);
        
        HighsInt Highs_getHighsIntInfoValue(const void* highs, const char* info,
                                            HighsInt* value);
        
        HighsInt Highs_getHighsDoubleInfoValue(const void* highs, const char* info,
                                               double* value);
        
        HighsInt Highs_getNumCols(const void* highs);
        
        HighsInt Highs_getNumRows(const void* highs);
        
        double Highs_getHighsInfinity(const void* highs);
        
        double Highs_getHighsRunTime(const void* highs);
        
        HighsInt Highs_setOptionValue(void* highs, const char* option,
                                      const char* value);
        
        HighsInt Highs_getScaledModelStatus(const void* highs);
    """
    )

    STATUS_ERROR = highslib.kHighsStatusError


def check(status):
    if status == STATUS_ERROR:
        raise mip.InterfacingError("Unknown error in call to HiGHS.")


class SolverHighs(mip.Solver):
    def __init__(self, model: mip.Model, name: str, sense: str):
        if not has_highs:
            raise FileNotFoundError(
                "HiGHS not found."
                "Please install the `highspy` package, or"
                "set the `PMIP_HIGHS_LIBRARY` environment variable."
            )

        # Store reference to library so that it's not garbage-collected (when we
        # just use highslib in __del__, it had already become None)?!
        self._lib = highslib

        super().__init__(model, name, sense)

        # Model creation and initialization.
        self._model = highslib.Highs_create()
        self.set_objective_sense(sense)

        # Store additional data here, if HiGHS can't do it.
        self._name: str = name
        self._num_int_vars = 0

        # Also store solution (when available)
        self._x = []
        self._rc = []
        self._pi = []

        # Buffer string for storing names
        self._name_buffer = ffi.new(f"char[{self._lib.kHighsMaximumStringLength}]")

        # type conversion maps
        self._var_type_map = {
            mip.CONTINUOUS: self._lib.kHighsVarTypeContinuous,
            mip.BINARY: self._lib.kHighsVarTypeInteger,
            mip.INTEGER: self._lib.kHighsVarTypeInteger,
        }
        self._highs_type_map = {value: key for key, value in self._var_type_map.items()}

    def __del__(self):
        self._name_buffer = None
        self._lib.Highs_destroy(self._model)

    def _get_int_info_value(self: "SolverHighs", name: str) -> int:
        value = ffi.new("int*")
        check(self._lib.Highs_getIntInfoValue(self._model, name.encode("UTF-8"), value))
        return value[0]

    def _get_double_info_value(self: "SolverHighs", name: str) -> float:
        value = ffi.new("double*")
        check(
            self._lib.Highs_getDoubleInfoValue(self._model, name.encode("UTF-8"), value)
        )
        return value[0]

    def _get_int_option_value(self: "SolverHighs", name: str) -> int:
        value = ffi.new("int*")
        check(
            self._lib.Highs_getIntOptionValue(self._model, name.encode("UTF-8"), value)
        )
        return value[0]

    def _get_double_option_value(self: "SolverHighs", name: str) -> float:
        value = ffi.new("double*")
        check(
            self._lib.Highs_getDoubleOptionValue(
                self._model, name.encode("UTF-8"), value
            )
        )
        return value[0]

    def _get_bool_option_value(self: "SolverHighs", name: str) -> float:
        value = ffi.new("bool*")
        check(
            self._lib.Highs_getBoolOptionValue(self._model, name.encode("UTF-8"), value)
        )
        return value[0]

    def _set_int_option_value(self: "SolverHighs", name: str, value: int):
        check(
            self._lib.Highs_setIntOptionValue(self._model, name.encode("UTF-8"), value)
        )

    def _set_double_option_value(self: "SolverHighs", name: str, value: float):
        check(
            self._lib.Highs_setDoubleOptionValue(
                self._model, name.encode("UTF-8"), value
            )
        )

    def _set_bool_option_value(self: "SolverHighs", name: str, value: float):
        check(
            self._lib.Highs_setBoolOptionValue(self._model, name.encode("UTF-8"), value)
        )

    def _change_coef(self: "SolverHighs", row: int, col: int, value: float):
        "Overwrite a single coefficient in the matrix."
        check(self._lib.Highs_changeCoeff(self._model, row, col, value))

    def _set_column(self: "SolverHighs", col: int, column: "mip.Column"):
        "Overwrite coefficients of one column."
        # We also have to set to 0 all coefficients of the old column, so we
        # fetch that first.
        var = self.model.vars[col]
        old_column = self.var_get_column(var)
        coeffs = {cons.idx: 0.0 for cons in old_column.constrs}
        coeffs.update(
            {cons.idx: coef for cons, coef in zip(column.constrs, column.coeffs)}
        )
        for row, coef in coeffs.items():
            self._change_coef(row, col, coef)

    def add_var(
        self: "SolverHighs",
        obj: numbers.Real = 0,
        lb: numbers.Real = 0,
        ub: numbers.Real = mip.INF,
        var_type: str = mip.CONTINUOUS,
        column: "mip.Column" = None,
        name: str = "",
    ):
        col: int = self.num_cols()
        check(self._lib.Highs_addCol(self._model, obj, lb, ub, 0, ffi.NULL, ffi.NULL))
        if name:
            check(self._lib.Highs_passColName(self._model, col, name.encode("utf-8")))
        if var_type != mip.CONTINUOUS:
            self._num_int_vars += 1
            check(
                self._lib.Highs_changeColIntegrality(
                    self._model, col, self._lib.kHighsVarTypeInteger
                )
            )

        if column:
            # Can't use _set_column here, because the variable is not added to
            # the mip.Model yet.
            # self._set_column(col, column)
            for cons, coef in zip(column.constrs, column.coeffs):
                self._change_coef(cons.idx, col, coef)

    def add_constr(self: "SolverHighs", lin_expr: "mip.LinExpr", name: str = ""):
        row: int = self.num_rows()

        # equation expressed as two-sided inequality
        lower = -lin_expr.const
        upper = -lin_expr.const
        if lin_expr.sense == mip.LESS_OR_EQUAL:
            lower = -mip.INF
        elif lin_expr.sense == mip.GREATER_OR_EQUAL:
            upper = mip.INF
        else:
            assert lin_expr.sense == mip.EQUAL

        num_new_nz = len(lin_expr.expr)
        index = ffi.new("int[]", [var.idx for var in lin_expr.expr.keys()])
        value = ffi.new("double[]", [coef for coef in lin_expr.expr.values()])

        check(
            self._lib.Highs_addRow(self._model, lower, upper, num_new_nz, index, value)
        )
        if name:
            self._lib.Highs_passRowName(self._model, row, name.encode("utf-8"))

    def add_lazy_constr(self: "SolverHighs", lin_expr: "mip.LinExpr"):
        raise NotImplementedError("HiGHS doesn't support lazy constraints!")

    def add_sos(
        self: "SolverHighs",
        sos: List[Tuple["mip.Var", numbers.Real]],
        sos_type: int,
    ):
        raise NotImplementedError("HiGHS doesn't support SOS!")

    def add_cut(self: "SolverHighs", lin_expr: "mip.LinExpr"):
        raise NotImplementedError("HiGHS doesn't support cut callbacks!")

    def get_objective_bound(self: "SolverHighs") -> numbers.Real:
        return self._get_double_info_value("mip_dual_bound")

    def get_objective(self: "SolverHighs") -> "mip.LinExpr":
        n = self.num_cols()
        num_col = ffi.new("int*")
        costs = ffi.new("double[]", n)
        lower = ffi.new("double[]", n)
        upper = ffi.new("double[]", n)
        num_nz = ffi.new("int*")
        check(
            self._lib.Highs_getColsByRange(
                self._model,
                0,  # from_col
                n - 1,  # to_col
                num_col,
                costs,
                lower,
                upper,
                num_nz,
                ffi.NULL,  # matrix_start
                ffi.NULL,  # matrix_index
                ffi.NULL,  # matrix_value
            )
        )
        obj_expr = mip.xsum(
            costs[i] * self.model.vars[i] for i in range(n) if costs[i] != 0.0
        )
        obj_expr.add_const(self.get_objective_const())
        obj_expr.sense = self.get_objective_sense()
        return obj_expr

    def get_objective_const(self: "SolverHighs") -> numbers.Real:
        offset = ffi.new("double*")
        check(self._lib.Highs_getObjectiveOffset(self._model, offset))
        return offset[0]

    def _all_cols_continuous(self: "SolverHighs"):
        n = self.num_cols()
        self._num_int_vars = 0
        integrality = ffi.new("int[]", [self._lib.kHighsVarTypeContinuous] * n)
        check(
            self._lib.Highs_changeColsIntegralityByRange(
                self._model, 0, n - 1, integrality
            )
        )

    def _reset_var_types(self: "SolverHighs", var_types: List[str]):
        integrality = ffi.new("int[]", [self._var_type_map[vt] for vt in var_types])
        n = self.num_cols()
        check(
            self._lib.Highs_changeColsIntegralityByRange(
                self._model, 0, n - 1, integrality
            )
        )
        self._num_int_vars = sum(1 for vt in var_types if vt != mip.CONTINUOUS)

    def relax(self: "SolverHighs"):
        # change integrality of all columns
        self._all_cols_continuous()

    def generate_cuts(
        self,
        cut_types: Optional[List[mip.CutType]] = None,
        depth: int = 0,
        npass: int = 0,
        max_cuts: int = mip.INT_MAX,
        min_viol: numbers.Real = 1e-4,
    ) -> "mip.CutPool":
        raise NotImplementedError("HiGHS doesn't support manual cut generation.")

    def clique_merge(self, constrs: Optional[List["mip.Constr"]] = None):
        raise NotImplementedError("HiGHS doesn't support clique merging!")

    def optimize(
        self: "SolverHighs",
        relax: bool = False,
    ) -> "mip.OptimizationStatus":
        if relax:
            # Temporarily change variable types.
            # Original types are stored in list var_type.
            var_types: List[str] = [var.var_type for var in self.model.vars]
            self._all_cols_continuous()

        self.set_mip_gap(self.model.max_mip_gap)
        self.set_mip_gap_abs(self.model.max_mip_gap_abs)

        check(self._lib.Highs_run(self._model))

        # check whether unsupported callbacks were set
        if self.model.lazy_constrs_generator:
            raise NotImplementedError(
                "HiGHS doesn't support lazy constraints at the moment"
            )
        if self.model.cuts_generator:
            raise NotImplementedError(
                "HiGHS doesn't support cuts generator at the moment"
            )

        # store solution values for later access
        opt_status = self.get_status()
        if opt_status in (
            mip.OptimizationStatus.OPTIMAL,
            mip.OptimizationStatus.FEASIBLE,
        ):
            n, m = self.num_cols(), self.num_rows()
            col_value = ffi.new("double[]", n)
            col_dual = ffi.new("double[]", n)
            row_value = ffi.new("double[]", m)
            row_dual = ffi.new("double[]", m)
            check(
                self._lib.Highs_getSolution(
                    self._model, col_value, col_dual, row_value, row_dual
                )
            )
            self._x = [col_value[j] for j in range(n)]
            self._rc = [col_dual[j] for j in range(n)]

            if self._has_dual_solution():
                self._pi = [row_dual[i] for i in range(m)]

        if relax:
            # Undo the temporary changes.
            self._reset_var_types(var_types)

        return opt_status

    def get_objective_value(self: "SolverHighs") -> numbers.Real:
        # only give value if we have stored a solution
        if self._x:
            return self._lib.Highs_getObjectiveValue(self._model)

    def get_log(
        self: "SolverHighs",
    ) -> List[Tuple[numbers.Real, Tuple[numbers.Real, numbers.Real]]]:
        raise NotImplementedError("HiGHS doesn't give access to a progress log.")

    def get_objective_value_i(self: "SolverHighs", i: int) -> numbers.Real:
        raise NotImplementedError("HiGHS doesn't store multiple solutions.")

    def get_num_solutions(self: "SolverHighs") -> int:
        # Multiple solutions are not supported (through C API?).
        return 1 if self._has_primal_solution() else 0

    def get_objective_sense(self: "SolverHighs") -> str:
        sense = ffi.new("int*")
        check(self._lib.Highs_getObjectiveSense(self._model, sense))
        sense_map = {
            self._lib.kHighsObjSenseMaximize: mip.MAXIMIZE,
            self._lib.kHighsObjSenseMinimize: mip.MINIMIZE,
        }
        return sense_map[sense[0]]

    def set_objective_sense(self: "SolverHighs", sense: str):
        sense_map = {
            mip.MAXIMIZE: self._lib.kHighsObjSenseMaximize,
            mip.MINIMIZE: self._lib.kHighsObjSenseMinimize,
        }
        check(self._lib.Highs_changeObjectiveSense(self._model, sense_map[sense]))

    def set_start(self: "SolverHighs", start: List[Tuple["mip.Var", numbers.Real]]):
        # using zeros for unset variables
        nvars = len(self.model.vars)
        cval = ffi.new("double[]", [0.0 for _ in range(nvars)])
        for col in start:
            cval[col[0].idx] = col[1]

        self._lib.Highs_setSolution(self._model, cval, ffi.NULL, ffi.NULL, ffi.NULL)

    def set_objective(self: "SolverHighs", lin_expr: "mip.LinExpr", sense: str = ""):
        # set coefficients
        for var, coef in lin_expr.expr.items():
            check(self._lib.Highs_changeColCost(self._model, var.idx, coef))

        self.set_objective_const(lin_expr.const)
        if lin_expr.sense:
            self.set_objective_sense(lin_expr.sense)

    def set_objective_const(self: "SolverHighs", const: numbers.Real):
        check(self._lib.Highs_changeObjectiveOffset(self._model, const))

    def set_processing_limits(
        self: "SolverHighs",
        max_time: numbers.Real = mip.INF,
        max_nodes: int = mip.INT_MAX,
        max_sol: int = mip.INT_MAX,
        max_seconds_same_incumbent: float = mip.INF,
        max_nodes_same_incumbent: int = mip.INT_MAX,
    ):
        if max_time != mip.INF:
            self.set_max_seconds(max_time)
        if max_nodes != mip.INT_MAX:
            self.set_max_nodes(max_nodes)
        if max_sol != mip.INT_MAX:
            self.set_max_solutions(max_sol)
        if max_seconds_same_incumbent != mip.INF:
            raise NotImplementedError("Can't set max_seconds_same_incumbent!")
        if max_nodes_same_incumbent != mip.INT_MAX:
            self.set_max_nodes_same_incumbent(max_nodes_same_incumbent)

    def get_max_seconds(self: "SolverHighs") -> numbers.Real:
        return self._get_double_option_value("time_limit")

    def set_max_seconds(self: "SolverHighs", max_seconds: numbers.Real):
        self._set_double_option_value("time_limit", max_seconds)

    def get_max_solutions(self: "SolverHighs") -> int:
        return self._get_int_option_value("mip_max_improving_sols")

    def set_max_solutions(self: "SolverHighs", max_solutions: int):
        self._set_int_option_value("mip_max_improving_sols", max_solutions)

    def get_pump_passes(self: "SolverHighs") -> int:
        raise NotImplementedError("HiGHS doesn't support pump passes.")

    def set_pump_passes(self: "SolverHighs", passes: int):
        raise NotImplementedError("HiGHS doesn't support pump passes.")

    def get_max_nodes(self: "SolverHighs") -> int:
        return self._get_int_option_value("mip_max_nodes")

    def set_max_nodes(self: "SolverHighs", max_nodes: int):
        self._set_int_option_value("mip_max_nodes", max_nodes)

    def get_max_nodes_same_incumbent(self: "SolverHighs") -> int:
        return self._get_int_option_value("mip_max_stall_nodes")

    def set_max_nodes_same_incumbent(self: "SolverHighs", max_nodes_same_incumbent: int):
        self._set_int_option_value("mip_max_stall_nodes", max_nodes_same_incumbent)

    def set_num_threads(self: "SolverHighs", threads: int):
        self._set_int_option_value("threads", threads)

    def write(self: "SolverHighs", file_path: str):
        check(self._lib.Highs_writeModel(self._model, file_path.encode("utf-8")))

    def read(self: "SolverHighs", file_path: str):
        if file_path.lower().endswith(".bas"):
            raise NotImplementedError("HiGHS does not support bas files")
        check(self._lib.Highs_readModel(self._model, file_path.encode("utf-8")))

    def num_cols(self: "SolverHighs") -> int:
        return self._lib.Highs_getNumCol(self._model)

    def num_rows(self: "SolverHighs") -> int:
        return self._lib.Highs_getNumRow(self._model)

    def num_nz(self: "SolverHighs") -> int:
        return self._lib.Highs_getNumNz(self._model)

    def num_int(self: "SolverHighs") -> int:
        return self._num_int_vars

    def get_emphasis(self: "SolverHighs") -> mip.SearchEmphasis:
        raise NotImplementedError("HiGHS doesn't support search emphasis.")

    def set_emphasis(self: "SolverHighs", emph: mip.SearchEmphasis):
        raise NotImplementedError("HiGHS doesn't support search emphasis.")

    def get_cutoff(self: "SolverHighs") -> numbers.Real:
        return self._get_double_option_value("objective_bound")

    def set_cutoff(self: "SolverHighs", cutoff: numbers.Real):
        self._set_double_option_value("objective_bound", cutoff)

    def get_mip_gap_abs(self: "SolverHighs") -> numbers.Real:
        return self._get_double_option_value("mip_abs_gap")

    def set_mip_gap_abs(self: "SolverHighs", mip_gap_abs: numbers.Real):
        self._set_double_option_value("mip_abs_gap", mip_gap_abs)

    def get_mip_gap(self: "SolverHighs") -> numbers.Real:
        return self._get_double_option_value("mip_rel_gap")

    def set_mip_gap(self: "SolverHighs", mip_gap: numbers.Real):
        self._set_double_option_value("mip_rel_gap", mip_gap)

    def get_verbose(self: "SolverHighs") -> int:
        return self._get_bool_option_value("output_flag")

    def set_verbose(self: "SolverHighs", verbose: int):
        self._set_bool_option_value("output_flag", verbose)

    # Constraint-related getters/setters

    def constr_get_expr(self: "SolverHighs", constr: "mip.Constr") -> "mip.LinExpr":
        row = constr.idx
        # Call method twice:
        #  - first, to get the sizes for coefficients,
        num_row = ffi.new("int*")
        lower = ffi.new("double[]", 1)
        upper = ffi.new("double[]", 1)
        num_nz = ffi.new("int*")
        # TODO: We also pass a non-NULL matrix_start, which should not be
        # needed, but works around a known bug in HiGHS' C API.
        _tmp_matrix_start = ffi.new("int[]", 1)
        check(
            self._lib.Highs_getRowsByRange(
                self._model,
                row,
                row,
                num_row,
                lower,
                upper,
                num_nz,
                _tmp_matrix_start,
                ffi.NULL,
                ffi.NULL,
            )
        )

        #  - second, to get the coefficients in pre-allocated arrays.
        if num_nz[0] == 0:
            # early exit for empty expressions
            expr = mip.xsum([])
        else:
            matrix_start = ffi.new("int[]", 1)
            matrix_index = ffi.new("int[]", num_nz[0])
            matrix_value = ffi.new("double[]", num_nz[0])
            check(
                self._lib.Highs_getRowsByRange(
                    self._model,
                    row,
                    row,
                    num_row,
                    lower,
                    upper,
                    num_nz,
                    matrix_start,
                    matrix_index,
                    matrix_value,
                )
            )
            expr = mip.xsum(
                matrix_value[i] * self.model.vars[matrix_index[i]]
                for i in range(num_nz[0])
            )

        # Also set sense and constant
        lhs, rhs = lower[0], upper[0]
        if rhs < mip.INF:
            expr -= rhs
            if lhs > -mip.INF:
                assert lhs == rhs
                expr.sense = mip.EQUAL
            else:
                expr.sense = mip.LESS_OR_EQUAL
        else:
            if lhs > -mip.INF:
                expr -= lhs
                expr.sense = mip.GREATER_OR_EQUAL
            else:
                raise ValueError("Unbounded constraint?!")
        return expr

    def constr_set_expr(
        self: "SolverHighs", constr: "mip.Constr", value: "mip.LinExpr"
    ) -> "mip.LinExpr":
        # We also have to set to 0 all coefficients of the old row, so we
        # fetch that first.
        coeffs = {var: 0.0 for var in constr.expr}

        # Then we fetch the new coefficients and overwrite.
        coeffs.update(value.expr.items())

        # Finally, we change the coeffs in HiGHS' matrix one-by-one.
        for var, coef in coeffs.items():
            self._change_coef(constr.idx, var.idx, coef)

    def constr_get_rhs(self: "SolverHighs", idx: int) -> numbers.Real:
        # fetch both lower and upper bound
        num_row = ffi.new("int*")
        lower = ffi.new("double[]", 1)
        upper = ffi.new("double[]", 1)
        num_nz = ffi.new("int*")
        check(
            self._lib.Highs_getRowsByRange(
                self._model,
                idx,
                idx,
                num_row,
                lower,
                upper,
                num_nz,
                ffi.NULL,
                ffi.NULL,
                ffi.NULL,
            )
        )

        # case distinction for sense
        if lower[0] == -mip.INF:
            return upper[0]
        if upper[0] == mip.INF:
            return lower[0]
        assert lower[0] == upper[0]
        return lower[0]

    def constr_set_rhs(self: "SolverHighs", idx: int, rhs: numbers.Real):
        # first need to figure out which bound to change (lower or upper)
        num_row = ffi.new("int*")
        lower = ffi.new("double[]", 1)
        upper = ffi.new("double[]", 1)
        num_nz = ffi.new("int*")
        check(
            self._lib.Highs_getRowsByRange(
                self._model,
                idx,
                idx,
                num_row,
                lower,
                upper,
                num_nz,
                ffi.NULL,
                ffi.NULL,
                ffi.NULL,
            )
        )

        # update bounds as needed
        lb, ub = lower[0], upper[0]
        if lb != -mip.INF:
            lb = rhs
        if ub != mip.INF:
            ub = rhs

        # set new bounds
        check(self._lib.Highs_changeRowBounds(self._model, idx, lb, ub))

    def constr_get_name(self: "SolverHighs", idx: int) -> str:
        name = self._name_buffer
        check(self._lib.Highs_getRowName(self._model, idx, name))
        return ffi.string(name).decode("utf-8")

    def constr_get_pi(self: "SolverHighs", constr: "mip.Constr") -> numbers.Real:
        if self._pi:
            return self._pi[constr.idx]

    def constr_get_slack(self: "SolverHighs", constr: "mip.Constr") -> numbers.Real:
        expr = constr.expr
        activity = sum(coef * var.x for var, coef in expr.expr.items())
        rhs = -expr.const
        slack = rhs - activity
        if expr.sense == mip.LESS_OR_EQUAL:
            return slack
        elif expr.sense == mip.GREATER_OR_EQUAL:
            return -slack
        elif expr.sense == mip.EQUAL:
            return -abs(slack)
        else:
            raise ValueError(f"Invalid constraint sense: {expr.sense}")

    def remove_constrs(self: "SolverHighs", constrsList: List[int]):
        set_ = ffi.new("int[]", constrsList)
        check(self._lib.Highs_deleteRowsBySet(self._model, len(constrsList), set_))

    def constr_get_index(self: "SolverHighs", name: str) -> int:
        idx = ffi.new("int *")
        self._lib.Highs_getRowByName(self._model, name.encode("utf-8"), idx)
        return idx[0]

    # Variable-related getters/setters

    def var_get_branch_priority(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        # TODO: Is actually not supported by HiGHS, but we mimic the behavior of
        # CBC and simply pretend that it's always 0.
        return 0

    def var_set_branch_priority(
        self: "SolverHighs", var: "mip.Var", value: numbers.Real
    ):
        # TODO: better raise warning/error instead?
        pass

    def var_get_lb(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        num_col = ffi.new("int*")
        costs = ffi.new("double[]", 1)
        lower = ffi.new("double[]", 1)
        upper = ffi.new("double[]", 1)
        num_nz = ffi.new("int*")
        check(
            self._lib.Highs_getColsByRange(
                self._model,
                var.idx,  # from_col
                var.idx,  # to_col
                num_col,
                costs,
                lower,
                upper,
                num_nz,
                ffi.NULL,  # matrix_start
                ffi.NULL,  # matrix_index
                ffi.NULL,  # matrix_value
            )
        )
        return lower[0]

    def var_set_lb(self: "SolverHighs", var: "mip.Var", value: numbers.Real):
        # can only set both bounds, so we just set the old upper bound
        old_upper = self.var_get_ub(var)
        check(self._lib.Highs_changeColBounds(self._model, var.idx, value, old_upper))

    def var_get_ub(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        num_col = ffi.new("int*")
        costs = ffi.new("double[]", 1)
        lower = ffi.new("double[]", 1)
        upper = ffi.new("double[]", 1)
        num_nz = ffi.new("int*")
        check(
            self._lib.Highs_getColsByRange(
                self._model,
                var.idx,  # from_col
                var.idx,  # to_col
                num_col,
                costs,
                lower,
                upper,
                num_nz,
                ffi.NULL,  # matrix_start
                ffi.NULL,  # matrix_index
                ffi.NULL,  # matrix_value
            )
        )
        return upper[0]

    def var_set_ub(self: "SolverHighs", var: "mip.Var", value: numbers.Real):
        # can only set both bounds, so we just set the old lower bound
        old_lower = self.var_get_lb(var)
        check(self._lib.Highs_changeColBounds(self._model, var.idx, old_lower, value))

    def var_get_obj(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        num_col = ffi.new("int*")
        costs = ffi.new("double[]", 1)
        lower = ffi.new("double[]", 1)
        upper = ffi.new("double[]", 1)
        num_nz = ffi.new("int*")
        check(
            self._lib.Highs_getColsByRange(
                self._model,
                var.idx,  # from_col
                var.idx,  # to_col
                num_col,
                costs,
                lower,
                upper,
                num_nz,
                ffi.NULL,  # matrix_start
                ffi.NULL,  # matrix_index
                ffi.NULL,  # matrix_value
            )
        )
        return costs[0]

    def var_set_obj(self: "SolverHighs", var: "mip.Var", value: numbers.Real):
        check(self._lib.Highs_changeColCost(self._model, var.idx, value))

    def var_get_var_type(self: "SolverHighs", var: "mip.Var") -> str:
        var_type = ffi.new("int*")
        ret = self._lib.Highs_getColIntegrality(self._model, var.idx, var_type)
        if var_type[0] not in self._highs_type_map:
            raise ValueError(
                f"Invalid variable type returned by HiGHS: {var_type[0]} (ret={ret})"
            )
        return self._highs_type_map[var_type[0]]

    def var_set_var_type(self: "SolverHighs", var: "mip.Var", value: str):
        if value not in self._var_type_map:
            raise ValueError(f"Invalid variable type: {value}")
        prev_var_type = var.var_type
        if value != prev_var_type:
            check(
                self._lib.Highs_changeColIntegrality(
                    self._model, var.idx, self._var_type_map[value]
                )
            )
            if prev_var_type != mip.CONTINUOUS and value == mip.CONTINUOUS:
                self._num_int_vars -= 1
            elif prev_var_type == mip.CONTINUOUS and value != mip.CONTINUOUS:
                self._num_int_vars += 1

    def var_get_column(self: "SolverHighs", var: "mip.Var") -> "mip.Column":
        # Call method twice:
        #  - first, to get the sizes for coefficients,
        num_col = ffi.new("int*")
        costs = ffi.new("double[]", 1)
        lower = ffi.new("double[]", 1)
        upper = ffi.new("double[]", 1)
        num_nz = ffi.new("int*")
        check(
            self._lib.Highs_getColsByRange(
                self._model,
                var.idx,  # from_col
                var.idx,  # to_col
                num_col,
                costs,
                lower,
                upper,
                num_nz,
                ffi.NULL,  # matrix_start
                ffi.NULL,  # matrix_index
                ffi.NULL,  # matrix_value
            )
        )
        #  - second, to get the coefficients in pre-allocated arrays.
        matrix_start = ffi.new("int[]", 1)
        matrix_index = ffi.new("int[]", num_nz[0])
        matrix_value = ffi.new("double[]", num_nz[0])
        check(
            self._lib.Highs_getColsByRange(
                self._model,
                var.idx,  # from_col
                var.idx,  # to_col
                num_col,
                costs,
                lower,
                upper,
                num_nz,
                matrix_start,
                matrix_index,
                matrix_value,
            )
        )

        return mip.Column(
            constrs=[self.model.constrs[matrix_index[i]] for i in range(num_nz[0])],
            coeffs=[matrix_value[i] for i in range(num_nz[0])],
        )

    def var_set_column(self: "SolverHighs", var: "mip.Var", value: "mip.Column"):
        self._set_column(var.idx, value)

    def var_get_rc(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        if self._rc:
            return self._rc[var.idx]

    def var_get_x(self: "SolverHighs", var: "mip.Var") -> numbers.Real:
        if self._x:
            return self._x[var.idx]

    def var_get_xi(self: "SolverHighs", var: "mip.Var", i: int) -> numbers.Real:
        raise NotImplementedError("HiGHS doesn't store multiple solutions.")

    def var_get_name(self: "SolverHighs", idx: int) -> str:
        name = self._name_buffer
        check(self._lib.Highs_getColName(self._model, idx, name))
        return ffi.string(name).decode("utf-8")

    def remove_vars(self: "SolverHighs", varsList: List[int]):
        set_ = ffi.new("int[]", varsList)
        check(self._lib.Highs_deleteColsBySet(self._model, len(varsList), set_))

    def var_get_index(self: "SolverHighs", name: str) -> int:
        idx = ffi.new("int *")
        self._lib.Highs_getColByName(self._model, name.encode("utf-8"), idx)
        return idx[0]

    def get_problem_name(self: "SolverHighs") -> str:
        return self._name

    def set_problem_name(self: "SolverHighs", name: str):
        self._name = name

    def _get_primal_solution_status(self: "SolverHighs"):
        return self._get_int_info_value("primal_solution_status")

    def _has_primal_solution(self: "SolverHighs"):
        return (
            self._get_primal_solution_status() == self._lib.kHighsSolutionStatusFeasible
        )

    def _get_dual_solution_status(self: "SolverHighs"):
        return self._get_int_info_value("dual_solution_status")

    def _has_dual_solution(self: "SolverHighs"):
        return self._get_dual_solution_status() == self._lib.kHighsSolutionStatusFeasible

    def get_status(self: "SolverHighs") -> mip.OptimizationStatus:
        OS = mip.OptimizationStatus
        status_map = {
            self._lib.kHighsModelStatusNotset: OS.OTHER,
            self._lib.kHighsModelStatusLoadError: OS.ERROR,
            self._lib.kHighsModelStatusModelError: OS.ERROR,
            self._lib.kHighsModelStatusPresolveError: OS.ERROR,
            self._lib.kHighsModelStatusSolveError: OS.ERROR,
            self._lib.kHighsModelStatusPostsolveError: OS.ERROR,
            self._lib.kHighsModelStatusModelEmpty: OS.OTHER,
            self._lib.kHighsModelStatusOptimal: OS.OPTIMAL,
            self._lib.kHighsModelStatusInfeasible: OS.INFEASIBLE,
            self._lib.kHighsModelStatusUnboundedOrInfeasible: OS.UNBOUNDED,
            # ... or should it be INFEASIBLE?
            self._lib.kHighsModelStatusUnbounded: OS.UNBOUNDED,
            self._lib.kHighsModelStatusObjectiveBound: None,
            self._lib.kHighsModelStatusObjectiveTarget: None,
            self._lib.kHighsModelStatusTimeLimit: None,
            self._lib.kHighsModelStatusIterationLimit: None,
            self._lib.kHighsModelStatusUnknown: OS.OTHER,
            self._lib.kHighsModelStatusSolutionLimit: None,
        }
        highs_status = self._lib.Highs_getModelStatus(self._model)
        status = status_map[highs_status]
        if status is None:
            # depends on solution status
            status = OS.FEASIBLE if self._has_primal_solution() else OS.NO_SOLUTION_FOUND
        return status

    def cgraph_density(self: "SolverHighs") -> float:
        """Density of the conflict graph"""
        raise NotImplementedError("HiGHS doesn't support conflict graph.")

    def conflicting(
        self: "SolverHighs",
        e1: Union["mip.LinExpr", "mip.Var"],
        e2: Union["mip.LinExpr", "mip.Var"],
    ) -> bool:
        """Checks if two assignment to binary variables are in conflict,
        returns none if no conflict graph is available"""
        raise NotImplementedError("HiGHS doesn't support conflict graph.")

    def conflicting_nodes(
        self: "SolverHighs", v1: Union["mip.Var", "mip.LinExpr"]
    ) -> Tuple[List["mip.Var"], List["mip.Var"]]:
        """Returns all assignment conflicting with the assignment in v1 in the
        conflict graph.
        """
        raise NotImplementedError("HiGHS doesn't support conflict graph.")

    def feature_values(self: "SolverHighs") -> List[float]:
        raise NotImplementedError("HiGHS doesn't support feature extraction.")

    def feature_names(self: "SolverHighs") -> List[str]:
        raise NotImplementedError("HiGHS doesn't support feature extraction.")
