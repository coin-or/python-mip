.. _chapInstall:

Installation
============

Python-MIP requires Python 3.5 or newer. Since Python-MIP is included in the `Python Package Index <https://pypi.org>`_, once you have a `Python installation <https://www.python.org/downloads>`_, installing it is as easy as entering in the command prompt:

.. code-block:: sh

    pip install mip

If the command fails, it may be due to lack of permission to install globally available Python modules. In this case, use:

.. code-block:: sh

    pip install mip --user

The default installation includes pre-compiled libraries of the MIP Solver `CBC <https://projects.coin-or.org/Cbc>`_ for Windows, Linux and MacOS.
If you have the commercial solver `Gurobi <http://gurobi.com>`_ installed in your computer, Python-MIP will automatically use it as long as it finds the Gurobi dynamic loadable library. Gurobi is free for academic use and has an outstanding performance for solving MIPs. Instructions to make it accessible on different operating systems are included bellow.


Gurobi Installation and Configuration (optional)
------------------------------------------------

For the installation of Gurobi you can look at the `Quickstart guide <https://www.gurobi.com/documentation/quickstart.html>`_ for your operating system. Python-MIP will automatically find your Gurobi installation as long as you define the :code:`GUROBI_HOME` environment variable indicating where Gurobi was installed.

Pypy installation (optional)
----------------------------

Python-MIP is compatible with the just-in-time Python compiler `Pypy <https://pypy.org>`_.
Generally, Python code executes much faster in Pypy.
Pypy is also more memory efficient.
To install Python-MIP as a Pypy package, just call (add :code:`--user` may be necessary also):

.. code-block:: sh

    pypy3 -m pip install mip

