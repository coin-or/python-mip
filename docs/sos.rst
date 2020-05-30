.. _chapSOS:

Special Ordered Sets 
====================

Special Ordered Sets (SOSs) are ordered sets of variables, where only one/two
contiguous variables in this set can assume non-zero values. Introduced in [BeTo70]_, they provide powerful means of modeling nonconvex functions [BeFo76]_ and can improve the performance of the branch-and-bound algorithm.

Type 1 SOS (S1):
    In this case, only one variable of the set can assume a non zero value. This variable may indicate, for example the site where a plant should be build. As the value of this non-zero variable would not have to be necessarily its upper bound, its value may also indicate the size of the plant.

Type 2 SOS (S2):
    In this case, up to two consecutive variables in the set may assume non-zero values. S2 are specially useful to model piecewise linear approximations of non-linear functions.

Given nonlinear function :math:`f(x)`, a linear approximation can be computed for a set of :math:`k` points :math:`x_1, x_2, \ldots, x_k`, using continuous variables :math:`w_1, w_2, \ldots, w_k`, with the following constraints:

.. math::

    \sum_{i=1}^{k} w_i           & = 1  \\
    \sum_{i=1}^{k} x_i \ldotp w_i & = x


Thus, the result of :math:`f(x)` can be approximate in :math:`z`:

.. math::

   z = \sum_{i=1}^{k} f(x_i) \ldotp w_i


Provided that at most two of the :math:`w_i` variables are allowed to be non-zero and they are adjacent, which can be ensured by adding the pairs (variables, weight) :math:`\{(w_i, x_i) \forall i \in \{1,\ldots, k\}\}` to the model as a S2 set, using function :meth:`~mip.Model.add_sos`. The approximation is exact at the selected points and is adequately approximated by linear interpolation between them.

As an example, consider that the production cost of some product that due to some economy of scale phenomenon, is :math:`f(x) = 1520 * \log x`. The graph bellow depicts the growing of :math:`f(x)` for :math:`x \in [0, 150]`. Triangles indicate selected discretization points for :math:`x`. Observe that, in this case, the approximation (straight lines connecting the triangles) remains pretty close to the real curve using only 5 discretization points. Additional discretization points can be included, not necessarily evenly distributed, for an improved precision.

.. image:: ./images/log_cost.*
   :width: 60%
   :align: center

In this example, the approximation of :math:`z = 1520 \log x` for points :math:`x = (0, 10, 30, 70, 150)`, which correspond to :math:`z=(0, 3499.929, 5169.82, 6457.713, 7616.166)` could be computed with the following constraints over :math:`x, z` and :math:`w_1, \ldots, w_5` : 

.. math::

   w_1 + w_2 + w_3 + w_4 + w_5 = 1 \\
   x = 0 w_1 + 10 w_2 + 30 w_3 + 70 w_4 + 150 w_5 \\
   z = 0 w_1 + 3499.929 w_2 + 5169.82 w_3 + 6457.713 w_4 + 7616.166 w_5

provided that :math:`\{(w_1, 0),  (w_2, 10), (w_3, 30), (w_4, 70), (w_5, 150)\}` is included as S2.


For a complete example showing the use of Type 1 and Type 2 SOS see :ref:`this example <exSOS>`.

    
