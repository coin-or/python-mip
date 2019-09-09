.. _chapInstall:

Installation
============

Python-MIP requires Python 3.5 or newer. Since Python-MIP is included in the `Python Package Index <https://pypi.org>`_, once you have a `Python installation <https://www.python.org/downloads>`_, installing it is as easy as entering, in the command prompt:

.. code-block:: sh

    pip install mip

If the command fails, it may be due to lack of permission to install globally available Python modules.
In this case, use:

.. code-block:: sh

    pip install mip --user

The default installation includes re-compiled libraries of the MIP Solver `CBC <https://projects.coin-or.org/Cbc>`_ for Windows, Linux and MacOS.
If you have the commercial solver `Gurobi <http://gurobi.com>`_ installed in your computer, Python-MIP will automatically use it as long as it finds the Gurobi dynamic loadable library. Gurobi is free for academic use and has an outstanding performance for solving MIPs.
Instructions to make it accessible on different operating systems are included bellow.


Gurobi Installation and Configuration (optional)
------------------------------------------------

Linux
~~~~~

Linux Gurobi installation notes are available `here <http://www.gurobi.com/documentation/current/quickstart_linux.pdf>`_.
In Linux, the Gurobi dynamic loadable library file is :code:`libgurobixx.so`, where :code:`xx` stands for Gurobi's version.
You must add the library installation directory to the :code:`/etc/ld.so.conf` file and call :code:`ldconfig` after to make the library visible for all applications.
If Gurobi was installed in :code:`/opt/gurobi810` then you would have to add :code:`/opt/gurobi810/linux64/lib/` to :code:`/etc/ld.so.conf`.
Since this is a system wide configuration file, you will require super user permission to modify it.
To add this Path as a configuration only for your user account, thus not requiring super user privileges, enter this command before re-starting your session:

.. code-block:: sh

    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/gurobi810/linux64/lib/' >> ~/.profile


Windows
~~~~~~~

Windows Gurobi installation notes are available `here <http://www.gurobi.com/documentation/current/quickstart_windows.pdf>`_.
In Windows, Gurobi's dynamically loadable library is the file :code:`gurobixx.dll`, where :code:`xx` stands for Gurobi's version.
Be sure to set your `Path environment variable <https://www.computerhope.com/issues/ch000549.htm>`_ to include the installation folder of this file.

Pypy installation (optional)
----------------------------

Python-MIP is compatible with the just-in-time Python compiler `Pypy <https://pypy.org>`_.
Generally, Python code executes much faster in Pypy.
Pypy is also more memory efficient.
To install Python-MIP as a Pypy package, just call:

.. code-block:: sh

    pypy3 -m pip install mip

