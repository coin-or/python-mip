from mip.model import *
from ctypes import *
from ctypes.util import *
from typing import Dict


class SolverCbc(Solver):
    def __init__(self, model: Model, name: str, sense: str):
        super().__init__(model, name, sense)
        
        self._model = cbcNewModel()
        
        self._objconst = 0.0
                
        # setting objective sense
        if sense == MAXIMIZE:
            cbcSetObjSense(self._model, -1.0)


    def add_var(self,
                obj: float = 0,
                lb: float = 0,
                ub: float = float("inf"),
                coltype: str = "C",
                column: "Column" = None,
                name: str = "") -> int:
        # collecting column data
        numnz: c_int = 0 if column is None else len(column.constrs)
        vind: POINTER(c_int) = (c_int * numnz)()
        vval: POINTER(c_double) = (c_double * numnz)()
        
        # collecting column coefficients
        for i in range(numnz):
            vind[i] = column.constrs[i].idx
            vval[i] = column.coeffs[i]
        
        isInt : c_char = \
                c_char(1) if coltype.upper() == "B" or coltype.upper() == "I" \
                else c_char(0)
        
        idx : int = int(cbcNumCols(self._model))
        
        cbcAddCol(self._model, c_str(name), 
                c_double(lb), c_double(ub), c_double(obj),
                isInt, numnz, vind, vval )
        

        return idx


    def get_objective_const(self) -> float:
        return self._objconst


    def set_objective(self, lin_expr: "LinExpr", sense: str = "") -> None:
        # collecting variable coefficients
        for var, coeff in lin_expr.expr.items():
            cbcSetObjCoeff(self._model, var.idx, coeff)

        # objective function constant
        self._objconst = c_double(lin_expr.const)
        
        # setting objective sense
        if sense == MAXIMIZE:
            cbcSetObjSense(self._model, -1.0)
        elif sense == MINIMIZE:
            cbcSetObjSense(self._model, 1.0)


    def optimize(self) -> int:
        res : int = cbcSolve(self._model)
        
        if cbcIsAbandoned(self._model):
            return ERROR
        
        if cbcIsProvenOptimal(self._model):
            return OPTIMAL
        
        if cbcIsProvenInfeasible(self._model):
            return INFEASIBLE
        
        if cbcIsContinuousUnbounded(self._model):
            return UNBOUNDED
        
        if cbcNumIntegers(self._model):
            if cbcBestSolution(self._model):
                return FEASIBLE
        
        return INFEASIBLE
    
    
    def get_objective_value(self) -> float:
        return cbcObjValue(self._model)


    def var_get_x(self, var: Var) -> float:
        if cbcNumIntegers(self._model)>0:
            x = cbcBestSolution(self._model)
            if x == c_void_p(0):
                raise Exception('no solution found')
            return float(x[var.idx])
        else:
            x = cbcColSolution(self._model)
            return float(x[var.idx])


    def var_get_lb(self, var: "Var") -> float:
        res : float = float(cbcGetColLower(self._model)[var.idx])
        return res


    def var_get_name(self, idx : int) -> str:
        nameSpace : c_char_p = create_string_buffer(256)
        cbcGetColName(self._model, c_int(idx), nameSpace, 255)
        return namesSpace.value


    def add_constr(self, lin_expr: "LinExpr", name: str = "") -> int:
        # collecting linear expression data
        numnz: c_int = len(lin_expr.expr)
        cind: POINTER(c_int) = (c_int * numnz)()
        cval: POINTER(c_double) = (c_double * numnz)()
        
        # collecting variable coefficients
        for i, (var, coeff) in enumerate(lin_expr.expr.items()):
            cind[i] = var.idx
            cval[i] = coeff
        
        # constraint sense and rhs
        sense: c_char = c_char(ord(lin_expr.sense))
        rhs: c_double = c_double(-lin_expr.const)
        
        # constraint index
        idx : int = int(cbcNumRows(self._model))
        
        cbcAddRow( self._model, c_str(name), numnz, cind, cval, sense, rhs )

        return idx


    def write(self, file_path: str):
        if ".mps" in file_path.lower():
            cbcWriteMps(self._model, c_str(file_path))
        else:
            cbcWriteLp(self._model, c_str(file_path))


    def read(self, file_path: str) -> None:
        if ".mps" in file_path.lower():
            cbcReadMps(self._model, c_str(file_path))
        else:
            cbcReadLp(self._model, c_str(file_path))


    def num_cols(self) -> int:
        return cbcNumCols(self._model)


    def num_rows(self) -> int:
        return cbcNumRows(self._model)


    def constr_get_expr(self, constr: Constr) -> LinExpr:
        numnz = cbcGetRowNz(self._model, constr.idx)
        
        ridx = cbcGetRowIndices(self._model, constr.idx)
        rcoef = cbcGetRowCoeffs(self._model, constr.idx)
        
        rhs = cbcGetRowRHS(self._model, constr.idx)
        rsense = cbcGetRowSense(self._model, constr.idx).decode('utf-8').upper()
        
        sense = ''
        if (rsense == 'E'):
            sense = EQUAL
        elif (rsense == 'L'):
            sense = LESS_OR_EQUAL
        elif (rsense == 'G'):
            sense = GREATER_OR_EQUAL
        else:
            raise Exception('Unknow sense: {}'.format(rsense))
        
        expr = LinExpr(const=-rhs, sense=sense)
        for i in range(numnz):
            expr.add_var(self.model.vars[ridx[i]], rcoef[i])
        
        return expr

    def constr_get_name(self, idx : int) -> str:
        nameSpace : c_char_p = create_string_buffer(256)
        cbcGetRowName(self._model, c_int(idx), nameSpace, 255)
        return nameSpace.value
    
    def set_processing_limits(self,
        maxTime  = inf,
        maxNodes = inf,
        maxSol = inf ):
        m = self._model
        if maxTime != inf:
            cbcSetParameter( m, c_str('timeMode'), c_str('elapsed'))
            cbcSetParameter( m, c_str('seconds'), c_str('{}'.format(maxTime)))
        if maxNodes != inf:
            cbcSetParameter( m, c_str('maxNodes'), c_str('{}'.format(maxNodes)))
        if maxNodes != inf:
            cbcSetParameter( m, c_str('maxSolutions'), c_str('{}'.format(maxNodes)))


    def __del__(self):
        cbcDeleteModel(self._model)

has_cbc = False

try:
	try:
		# linux library
		cbclib = CDLL(find_library("CbcSolver"))
		has_cbc = True
		print('cbc found')
	except:
		# window library
		try:
			cbclib = CDLL(find_library("cbcCInterfaceDll"))
			has_cbc = True
			print('cbc found')
		except:
			cbclib = CDLL(find_library("./cbcCInterfaceDll"))
			has_cbc = True
			print('cbc found')
except: 
    has_cbc = False
    print('cbc not found')

if has_cbc:
	cbcNewModel = cbclib.Cbc_newModel
	cbcNewModel.restype = c_void_p

	cbcReadLp = cbclib.Cbc_readLp
	cbcReadLp.argtypes = [c_void_p, c_char_p]
	cbcReadLp.restype = c_int

	cbcReadMps = cbclib.Cbc_readMps
	cbcReadMps.argtypes = [c_void_p, c_char_p]
	cbcReadMps.restype = c_int

	cbcWriteLp = cbclib.Cbc_writeLp
	cbcWriteLp.argtypes = [c_void_p, c_char_p]

	cbcWriteMps = cbclib.Cbc_writeMps
	cbcWriteMps.argtypes = [c_void_p, c_char_p]

	cbcNumCols = cbclib.Cbc_getNumCols
	cbcNumCols.argtypes = [c_void_p]
	cbcNumCols.restype = c_int

	cbcNumIntegers = cbclib.Cbc_getNumIntegers
	cbcNumIntegers.argtypes = [c_void_p]
	cbcNumIntegers.restype = c_int

	cbcNumRows = cbclib.Cbc_getNumRows
	cbcNumRows.argtypes = [c_void_p]
	cbcNumRows.restype = c_int

	cbcGetRowNz = cbclib.Cbc_getRowNz
	cbcGetRowNz.argtypes = [c_void_p, c_int]
	cbcGetRowNz.restype = c_int

	cbcGetRowIndices = cbclib.Cbc_getRowIndices
	cbcGetRowIndices.argtypes = [c_void_p, c_int]
	cbcGetRowIndices.restype = POINTER(c_int)

	cbcGetRowCoeffs = cbclib.Cbc_getRowCoeffs
	cbcGetRowCoeffs.argtypes = [c_void_p, c_int]
	cbcGetRowCoeffs.restype = POINTER(c_double)

	cbcGetRowRHS = cbclib.Cbc_getRowRHS
	cbcGetRowRHS.argtypes = [c_void_p, c_int]
	cbcGetRowRHS.restype = c_double

	cbcGetRowSense = cbclib.Cbc_getRowSense
	cbcGetRowSense.argtypes = [c_void_p, c_int]
	cbcGetRowSense.restype = c_char

	cbcAddCol = cbclib.Cbc_addCol
	cbcAddCol.argtypes = [c_void_p, c_char_p, c_double, 
			c_double, c_double, c_char, c_int,
			POINTER(c_int), POINTER(c_double)]

	cbcAddRow = cbclib.Cbc_addRow
	cbcAddRow.argtypes = [c_void_p, c_char_p, c_int, 
			POINTER(c_int), POINTER(c_double), c_char, c_double]

	cbcSetObjCoeff = cbclib.Cbc_setObjCoeff
	cbcSetObjCoeff.argtypes = [c_void_p, c_int, c_double]

	cbcDeleteModel = cbclib.Cbc_deleteModel
	cbcDeleteModel.argtypes = [c_void_p]

	cbcSolve = cbclib.Cbc_solve
	cbcSolve.argtypes = [c_void_p]
	cbcSolve.restype = c_int

	cbcColSolution = cbclib.Cbc_getColSolution
	cbcColSolution.argtypes = [c_void_p]
	cbcColSolution.restype = POINTER(c_double)

	cbcBestSolution = cbclib.Cbc_bestSolution
	cbcBestSolution.argtypes = [c_void_p]
	cbcBestSolution.restype = POINTER(c_double)

	cbcObjValue = cbclib.Cbc_getObjValue
	cbcObjValue.argtypes = [c_void_p]
	cbcObjValue.restype = c_double

	cbcSetObjSense = cbclib.Cbc_setObjSense
	cbcSetObjSense.argtypes = [c_void_p, c_double]

	cbcIsProvenOptimal = cbclib.Cbc_isProvenOptimal
	cbcIsProvenOptimal.argtypes = [c_void_p]
	cbcIsProvenOptimal.restype = c_int

	cbcIsProvenInfeasible = cbclib.Cbc_isProvenInfeasible
	cbcIsProvenInfeasible.argtypes = [c_void_p]
	cbcIsProvenInfeasible.restype = c_int

	cbcIsContinuousUnbounded = cbclib.Cbc_isContinuousUnbounded
	cbcIsContinuousUnbounded.argtypes = [c_void_p]
	cbcIsContinuousUnbounded.restype = c_int

	cbcIsAbandoned = cbclib.Cbc_isAbandoned
	cbcIsAbandoned.argtypes = [c_void_p]
	cbcIsAbandoned.restype = c_int

	cbcGetColLower = cbclib.Cbc_getColLower
	cbcGetColLower.argtypes = [c_void_p]
	cbcGetColLower.restype = POINTER(c_double)

	cbcGetColUpper = cbclib.Cbc_getColUpper
	cbcGetColUpper.argtypes = [c_void_p]
	cbcGetColUpper.restype = POINTER(c_double)

	cbcSetColLower = cbclib.Cbc_setColLower
	cbcSetColLower.argtypes = [c_void_p, c_int, c_double]
	cbcSetColLower.restype = POINTER(c_double)

	cbcSetColUpper = cbclib.Cbc_setColUpper
	cbcSetColUpper.argtypes = [c_void_p, c_int, c_double]
	cbcSetColUpper.restype = POINTER(c_double)

	cbcGetColName = cbclib.Cbc_getColName
	cbcGetColName.argtypes = [c_void_p, c_int, c_char_p, c_int]

	cbcGetRowName = cbclib.Cbc_getRowName
	cbcGetRowName.argtypes = [c_void_p, c_int, c_char_p, c_int]

	cbcSetParameter = cbclib.Cbc_setParameter
	cbcSetParameter.argtypes = [c_void_p, c_char_p, c_char_p]

def c_str(value) -> c_char_p:
    """
    This function converts a python string into a C compatible char[]
    :param value: input string
    :return: string converted to C"s format
    """
    return create_string_buffer(value.encode("utf-8"))


