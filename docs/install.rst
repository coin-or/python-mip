.. _chapInstall:

Installation
============

Python-MIP requires Python 3.5 or newer. Since Python-MIP is included in
the `Python Package Index <https://pypi.org>`_, once you have a 
`Python installation <https://www.python.org/downloads>`_, 
installing it is as easy as entering: 

.. code-block:: sh
   
    pip install mip

in the command prompt. If this command fails, maybe you don't have permission
to install globally available Python modules. In this case, use:

.. code-block:: sh
   
    pip install mip --user

The default installation includes the open source
MIP Solver `CBC <https://projects.coin-or.org/Cbc>`_, which is used by default.
Pre-compiled CBC libraries for Windows, Linux and MacOS are shipped. If you
have the commercial solver `Gurobi <included>`_ installed in your computer
Python-MIP will automatically use it as long as it finds the Gurobi dynamic 
loadable library. Gurobi is free for academic use and has an outstanding performance
for solving hard MIPs.
Instructions to make it accessible on different operating 
systems are included bellow.


Gurobi Installation and Configuration (optional)
------------------------------------------------

Linux
~~~~~

Linux Gurobi installation notes are available 
`here <http://www.gurobi.com/documentation/current/quickstart_linux.pdf>`_. In Linux, the Gurobi 
dynamic loadable library file is :code:`libgurobixx.so`, where :code:`xx` is the Gurobi version. 
You must add the library installation directory to the :code:`/etc/ld.so.conf` file and call 
:code:`ldconfig` after to make the library visible for all applications. If Gurobi was installed in
:code:`/opt/gurobi810` then you would have to add :code:`/opt/gurobi810/linux64/lib/` to :code:`/etc/ld.so.conf`. 
Since this is a system wide configuration file, you will require super user permission to modify it. 
To add this Path as a configuration only for your user account, thus not requiring super user privileges, 
enter this command before re-starting your session:

.. code-block:: sh

    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/gurobi810/linux64/lib/' >> ~/.profile


Windows
~~~~~~~

Windows Gurobi installation notes are available 
`here <http://www.gurobi.com/documentation/current/quickstart_windows.pdf>`_. In
Windows, the Gurobi dynamically loadable library is the file :code:`gurobixx.dll`, where :code:`xx` 
is the Gurobi version. Be
sure to set your `Path environment variable
<https://www.computerhope.com/issues/ch000549.htm>`_ to the installation folder
of this file. 

You can execute your Python-MIP applications in the default Python
interpreter (`CPython <https://en.wikipedia.org/wiki/CPython>`_) or in the
`Pypy3 <https://pypy.org>`_ Just-in-Time compiler, which is much faster
and memory efficient.

Pypy installation (optional)
----------------------------

Python-MIP is compatible with the just-in-time Python compiler `Pypy <https://pypy.org>`_. 
Usually Python code executes much faster in Pypy. Pypy is also more memory efficient. To 
install Python-MIP as a Pypy package just call:

.. code-block:: sh

    pypy3 -m pip install mip

