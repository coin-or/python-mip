from mip import Model, OptimizationStatus


def test_windows_instability():
    model = Model()
    model.read("./data/windows-instability.lp")
    result = model.optimize(max_seconds=300)

    assert result in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE)
