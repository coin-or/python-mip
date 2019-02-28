Installation
============

Python-MIP requires Python 3.5 or newer. Since Python-MIP is included in
the Python Package Index, installing it is as easy as entering: 

.. code-block:: sh
   
    pip install mip

in the command prompt. The default installation includes the open source
MIP Solver `CBC <https://projects.coin-or.org/Cbc>`_, which is used by
default. Pre-compiled CBC libraries for Windows, Linux and MacOS are
shipped. If you have the commercial solver `Gurobi <included>`_ installed
in your computer Python-MIP will automatically use it [#f1]_. 

You can execute your Python-MIP applications in the default Python
interpreter (`CPython <https://en.wikipedia.org/wiki/CPython>`_) or in the
`Pypy3 <https://pypy.org>`_ Just-in-Time compiler, which is much faster
and memory efficient.

.. rubric:: Footnotes

.. [#f1] For Gurobi, be sure that the Gurobi dynamic library is reachable in
   the directories listed in the `PATH on Windows
   <https://www.computerhope.com/issues/ch000549.htm>`_ /`Library PATH on
   Linux
   <http://howtolamp.com/articles/adding-shared-libraries-to-system-library-path/>`_
   /MacOS.
