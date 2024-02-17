import pytest

import mip
import mip.gurobi


def test_gurobi_pip_installation():
    # Even though we have no valid license yet, we could check that the binaries are found.
    # If no valid license is found, an InterfacingError is thrown

    if mip.gurobi.has_gurobi:
        with pytest.raises(mip.InterfacingError):
            mip.Model(solver_name=mip.GUROBI)
