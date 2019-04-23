Developing Customized Branch-&-Cut algorithms
=============================================

Cut generators
~~~~~~~~~~~~~~

In many applications there are strong formulations that require an
exponential number of constraints. These formulations cannot be direct
handled by the MIP Solver: entering all these constraints at once is
usually not practical. Using cut generators you can interface with the MIP
solver so that at each node of the search tree you can insert only the
violated inequalities, called *cuts*. The problem of discovering these
violated inequalities is called the *Separation Problem*.
