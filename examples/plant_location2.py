r"""Example with Special Ordered Sets (SOS) type 1 and 2.
Plant location: one industry plans to install two plants in two different
regions, one to the west and another to the east. It must decide also the
production capacity of each plant and allocate clients to plants
in order to minimize shipping costs. We'll use SOS of type 1 to ensure that
only one of the plants in each region has a non-zero production capacity.
The cost :math:`f(z)` of building a plant with capacity :math:`z` grows according
to the non-linear function :math:`f(z)=1520 \log z`. Type 2 SOS will be used to
model the cost of installing each one of the plants.
"""

from __future__ import annotations

import sys

# Workaround for issues with python not being installed as a framework on mac
# by using a different backend.
if sys.platform == "darwin":  # OS X
    import matplotlib as mpl
    mpl.use('Agg')
    del mpl

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mip

if TYPE_CHECKING:
    from typing import Callable
    from collections.abc import Sequence


@dataclass(frozen=True)
class CandidateSite:
    """Potential factory locations."""
    location: tuple[float, float]
    max_capacity: float

@dataclass(frozen=True)
class Customer:
    location: tuple[float, float]
    demand: float

sites = (
    CandidateSite(location = ( 1, 38), max_capacity = 1955),
    CandidateSite(location = (31, 40), max_capacity = 1932),
    CandidateSite(location = (23, 59), max_capacity = 1987),
    CandidateSite(location = (76, 51), max_capacity = 1823),
    CandidateSite(location = (93, 51), max_capacity = 1718),
    CandidateSite(location = (63, 74), max_capacity = 1742),
)

customers = (
    Customer( location = (94, 10), demand = 302 ),
    Customer( location = (57, 26), demand = 273 ),
    Customer( location = (74, 44), demand = 275 ),
    Customer( location = (27, 51), demand = 266 ),
    Customer( location = (78, 30), demand = 287 ),
    Customer( location = (23, 30), demand = 296 ),
    Customer( location = (20, 72), demand = 297 ),
    Customer( location = ( 3, 27), demand = 310 ),
    Customer( location = ( 5, 39), demand = 302 ),
    Customer( location = (51,  1), demand = 309 ),
)

def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    return round(math.sqrt((ax-bx)**2 + (ay-by)**2), 1)

# pre-compute distances
_dist_table = {
    (site, cust) : _distance(site.location, cust.location)
    for site in sites
    for cust in customers
}

mip_model = mip.Model()

# Helper class
# SOS type 2 to approximate installation costs as piecewise linear function
class LinearApprox:
    """Approximate function by piecewise linear function.
    *x_pts* are the points in the domain used for the piecewise approximation."""
    def __init__(self, func: Callable[[float], float], x_pts: Sequence[float]):
        self.func = func
        self.x_pts = x_pts

    def __call__(self, x_expr: mip.LinExpr | mip.Var) -> mip.LinExpr:
        """Construct a linear expression that approximates the function at the given value,
        adding the necessary constraints."""
        model = x_expr.model

        # linear approximation points in the domain of F(x)
        x_pts = [ x for x in self.x_pts if float(x_expr.lb) <= x <= float(x_expr.ub) ]

        # non-linear function values for each approximation point
        y_pts = [ self.func(x) for x in x_pts ]

        # w variables - interpolation weights that add to 1, at most two can be non-zero
        w_vars = [ model.add_var() for _ in range(len(x_pts)) ]
        model.add_constr( mip.xsum(w_vars) == 1 )  # convexification
        model.add_sos([ (w, k) for w, k in zip(w_vars, x_pts) ], sos_type=2)

        # linear interpolate in each range to create function input and output expressions
        model.add_constr(mip.xsum(w * x for w, x in zip(w_vars, x_pts)) == x_expr)
        return mip.xsum(w * y for w, y in zip(w_vars, y_pts))

def _linspace(minv: float, maxv: float, pts: int) -> list[float]:
    return [ minv + (maxv - minv)*(n / (pts-1)) for n in range(pts) ]

class Factory:
    def __init__(self, model: mip.Model, site: CandidateSite):
        self.site = site

        # create a variable to decide how big to build each plant
        # zero size means the plant won't be built
        self.capacity = model.add_var(lb = 0, ub = site.max_capacity)

        # create a variable for build cost using an approximation of a non-linear function
        approx_pts = _linspace(float(self.capacity.lb), float(self.capacity.ub), 6)
        cost_approx = LinearApprox(self._build_cost, approx_pts)
        self.build_cost = cost_approx(self.capacity)

    @staticmethod
    def _build_cost(capacity: float) -> float:
        """Costs to build a factory with capacity `capacity`; nonlinear function"""
        # clamping the output to 0 if capacity < 1 (instead of capacity < 0) avoids a discontinuity
        return 1520 * math.log(capacity) if capacity > 1 else 0


factories = [
    Factory(mip_model, site) for site in sites
]

# Type 1 SOS: only one Factory per region
region1 = [ fact for fact in factories if  0 <= fact.site.location[0] <=  50 ]
region2 = [ fact for fact in factories if 50 <= fact.site.location[0] <= 100 ]

for region in (region1, region2):
    mip_model.add_sos([(fact.capacity, idx) for idx, fact in enumerate(region)], sos_type=1)


# Amount that each factory will supply each customer
_order_table = {
    (fact, cust) : mip_model.add_var()
    for fact in factories
    for cust in customers
}

# Constraint: satisfy demand
for cust in customers:
    mip_model.add_constr( mip.xsum(_order_table[fact, cust] for fact in factories) == cust.demand )

# Constraint: max factory production is limited by size
for fact in factories:
    mip_model.add_constr(mip.xsum(_order_table[fact, cust] for cust in customers ) <= fact.capacity)

# objective function
shipping_costs = mip.xsum(
    _order_table[fact, cust] * _dist_table[fact.site, cust]
    for fact in factories
    for cust in customers
)

build_costs = mip.xsum(fact.build_cost for fact in factories)

mip_model.objective = mip.minimize( shipping_costs + build_costs )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # plot possible plant locations
    for candidate in sites:
        x, y = candidate.location
        plt.scatter(x, y, marker="^", color="purple", s=50)
        plt.text(x, y, f"{candidate.max_capacity:d}")

    # plot location of customers
    for customer in customers:
        x, y = customer.location
        plt.scatter(x, y, marker="o", color="black", s=15)
        plt.text(x, y, f"{customer.demand:d}")

    plt.text(20, 78, "Region 1")
    plt.text(70, 78, "Region 2")
    plt.plot((50, 50), (0, 80))

    plt.savefig("location.pdf")

    mip_model.optimize()

    if mip_model.num_solutions:
        print(f"Solution with cost {mip_model.objective_value} found.")
        print(f"Facilities capacities: {[fact.capacity.x for fact in factories]}")
        print(f"Facilities cost: {[fact.build_cost.x for fact in factories]}")

        # plotting allocations
        for fact in factories:
            for cust in customers:
                if _order_table[fact, cust].x >= 1e-6:
                    xs, ys = zip(*[fact.site.location, cust.location])
                    plt.plot(xs, ys, linestyle="--", color="darkgray")

        plt.savefig("location-sol.pdf")

    # sanity checks
    opt = 99733.94905406
    if mip_model.status == mip.OptimizationStatus.OPTIMAL:
        assert abs(mip_model.objective_value - opt) <= 0.01
    elif mip_model.status == mip.OptimizationStatus.FEASIBLE:
        assert float(mip_model.objective_value) >= opt - 0.01
    else:
        assert mip_model.status not in (
            mip.OptimizationStatus.INFEASIBLE,
            mip.OptimizationStatus.UNBOUNDED,
        )
