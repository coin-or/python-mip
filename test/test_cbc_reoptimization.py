import pytest

from mip import BINARY, CBC, HIGHS, LP_Method, Model, OptimizationStatus
from mip.cbc import cbclib, ffi


def native_parameter(model, name):
    value = ffi.new("char[64]")
    assert cbclib.Cbc_getParam(model.solver._model, name.encode(), value, 64) == 0
    return ffi.string(value).decode()


@pytest.mark.parametrize(
    "method",
    [
        LP_Method.AUTO,
        LP_Method.DUAL,
        LP_Method.PRIMAL,
        LP_Method.BARRIER,
        LP_Method.RACING,
    ],
)
@pytest.mark.parametrize("presolve", [None, False, True])
def test_reoptimization_after_model_changes(method, presolve):
    m = Model(solver_name=CBC)
    m.verbose = 0
    m.threads = 2
    m.lp_method = LP_Method.DUAL
    m.lp_reopt_method = method
    m.lp_reopt_presolve = presolve
    m.lp_reopt_perturbation = 100
    x = m.add_var(obj=2)
    m += x >= 10
    assert m.optimize(relax=True) == OptimizationStatus.OPTIMAL
    assert m.objective_value == pytest.approx(20)

    y = m.add_var(obj=0.5)
    m += x + y >= 10
    m.remove(m.constrs[0])
    assert m.optimize(relax=True) == OptimizationStatus.OPTIMAL
    assert m.objective_value == pytest.approx(5)

    y.ub = 5
    assert m.optimize(relax=True) == OptimizationStatus.OPTIMAL
    assert m.objective_value == pytest.approx(12.5)
    assert native_parameter(m, "reoptMethod") == method.name.lower()
    assert native_parameter(m, "reoptPresolve") == (
        "auto" if presolve is None else "on" if presolve else "off"
    )
    assert native_parameter(m, "reoptPertValue") == "100"


def test_defaults_and_switching_reoptimization_parameters():
    m = Model(solver_name=CBC)
    m.verbose = 0
    assert m.lp_reopt_method == LP_Method.AUTO
    assert m.lp_reopt_presolve is None
    assert m.lp_reopt_perturbation == -1
    x = m.add_var(obj=1)
    m += x >= 1
    m.lp_reopt_method = LP_Method.PRIMAL
    m.lp_reopt_presolve = True
    m.lp_reopt_perturbation = 100
    assert m.optimize(relax=True) == OptimizationStatus.OPTIMAL
    m.lp_reopt_method = LP_Method.DUAL
    m.lp_reopt_presolve = None
    m.lp_reopt_perturbation = -1
    x.lb = 2
    assert m.optimize(relax=True) == OptimizationStatus.OPTIMAL
    assert m.objective_value == pytest.approx(2)
    assert native_parameter(m, "reoptMethod") == "dual"
    assert native_parameter(m, "reoptPresolve") == "auto"
    assert native_parameter(m, "reoptPertValue") == "-1"


@pytest.mark.parametrize(
    "name,value",
    [
        ("lp_reopt_method", LP_Method.RECOMMEND),
        ("lp_reopt_method", LP_Method.BARRIERNOCROSS),
        ("lp_reopt_method", "primal"),
        ("lp_reopt_presolve", 1),
        ("lp_reopt_presolve", "off"),
        ("lp_reopt_perturbation", True),
        ("lp_reopt_perturbation", 1.5),
        ("lp_reopt_perturbation", -5001),
        ("lp_reopt_perturbation", 103),
    ],
)
def test_invalid_reoptimization_parameters(name, value):
    m = Model(solver_name=CBC)
    previous = getattr(m, name)
    with pytest.raises(ValueError):
        setattr(m, name, value)
    assert getattr(m, name) == previous


@pytest.mark.parametrize("threads", [0, 1])
def test_racing_requires_threads_for_lp_and_mip(threads):
    m = Model(solver_name=CBC)
    m.verbose = 0
    m.threads = threads
    m.lp_reopt_method = LP_Method.RACING
    m.add_var(var_type=BINARY)
    with pytest.raises(ValueError, match="at least two threads"):
        m.optimize(relax=True)
    with pytest.raises(ValueError, match="at least two threads"):
        m.optimize()


def test_reoptimization_options_survive_read(tmp_path):
    path = tmp_path / "model.lp"
    path.write_text("Minimize\n obj: x\nSubject To\n c: x >= 1\nEnd\n")

    m = Model(solver_name=CBC)
    m.verbose = 0
    m.lp_reopt_method = LP_Method.PRIMAL
    m.lp_reopt_presolve = False
    m.lp_reopt_perturbation = 100
    m.read(str(path))
    assert m.optimize(relax=True) == OptimizationStatus.OPTIMAL
    assert native_parameter(m, "reoptMethod") == "primal"
    assert native_parameter(m, "reoptPresolve") == "off"
    assert native_parameter(m, "reoptPertValue") == "100"


def test_reoptimization_mip_solve():
    m = Model(solver_name=CBC)
    m.verbose = 0
    m.lp_reopt_method = LP_Method.PRIMAL
    m.lp_reopt_presolve = False
    m.lp_reopt_perturbation = 100
    x = m.add_var(var_type=BINARY, obj=-1)
    y = m.add_var(var_type=BINARY, obj=-1)
    m += 2 * x + 2 * y <= 3
    assert m.optimize() == OptimizationStatus.OPTIMAL
    assert m.objective_value == pytest.approx(-1)
    assert native_parameter(m, "reoptMethod") == "primal"
    assert native_parameter(m, "reoptPresolve") == "off"
    assert native_parameter(m, "reoptPertValue") == "100"


def test_other_solvers_reject_reoptimization_options():
    pytest.importorskip("highspy")
    m = Model(solver_name=HIGHS)
    with pytest.raises(NotImplementedError, match="only supported by CBC"):
        m.lp_reopt_method = LP_Method.PRIMAL
