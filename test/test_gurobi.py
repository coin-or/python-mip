import pytest

import mip


def test_gurobi_pip_installation():
    # Even though we have no valid license yet, we could check that the binaries are found.
    # If no valid license is found, an InterfacingError is thrown

    with pytest.raises(mip.InterfacingError):
        mip.Model(solver_name="GRB")

