from mip import Model, OptimizationStatus
from pathlib import Path


def test_windows_instability():
    model = Model()
    file = Path(__file__).parent / "data/windows-instability.lp"
    model.read(str(file.resolve()))
    result = model.optimize(max_seconds=300)

    assert result in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE)
