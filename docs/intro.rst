.. _chapIntro:

Introduction
============

The Python-MIP package provides tools for modeling and solving
`Mixed-Integer Linear Programming Problems
<https://en.wikipedia.org/wiki/Integer_programming>`_ (MIPs) [Wols98]_ in
Python. The default installation includes the `COIN-OR Linear Programming
Solver - CLP <http://github.com/coin-or/Clp>`_, which is currently the
`fastest <http://plato.asu.edu/ftp/lpsimp.html>`_  open source linear
programming solver and the `COIN-OR Branch-and-Cut solver - CBC
<https://github.com/coin-or/Cbc>`_, a highly configurable MIP solver. It
also works with the state-of-the-art `Gurobi <http://www.gurobi.com/>`_
MIP solver. Python-MIP was written in modern, `typed Python
<https://docs.python.org/3/library/typing.html>`_ and works with the fast
just-in-time Python compiler `Pypy <https://pypy.org/>`_. 

In the modeling layer, models can be written very concisely, as in high-level
mathematical programming languages such as `MathProg
<http://gusek.sourceforge.net/gmpl.pdf>`_. Modeling examples for some
applications can be viewed in :ref:`Chapter 4 <chapExamples>`.

Python-MIP eases the development of high-performance MIP based solvers for
custom applications by providing a tight integration with the
branch-and-cut algorithms of the supported solvers. Strong formulations
with an exponential number of constraints can be handled by the inclusion of
:ref:`Cut Generators <cut-generation-label>` and :ref:`Lazy Constraints <lazy-constraints-label>`.
Heuristics can be integrated for :ref:`providing initial feasible solutions
<mipstart-label>` to the MIP solver. These features can be used in both solver
engines, CBC and GUROBI, without changing a single line of code.

This document is organized as follows: in the :ref:`next Chapter
<chapInstall>` installation and configuration instructions for different
platforms are presented. In :ref:`Chapter 3 <chapQuick>` an overview of some
common model creation and optimization code included. Commented examples are included in
:ref:`Chapter 4 <chapExamples>`. :ref:`Chapter 5 <chapCustom>` includes
some common solver customizations that can be done to improve the
performance of application specific solvers. Finally, the detailed
reference information for the main classes is included in :ref:`Chapter
6 <chapClasses>`.

Releases and Source Code
------------------------

Python-MIP's source code is available at GitHub on (`github.com/coin-or/python-mip <https://github.com/coin-or/python-mip>`_). 
Releases may be downloaded/installed via PIP (see :ref:`next Chapter <chapInstall>`) or from our GitHub `Releases <https://github.com/coin-or/python-mip/releases>`_ page.

Getting help
------------

Questions, suggestions and feature request can be posted in our GitHub `Discussions <https://github.com/coin-or/python-mip/discussions>`_ page.

Acknowledgments
---------------

We would like to thank for the support of the `Combinatorial Optimization and Decision Support (CODeS) <https://set.kuleuven.be/codes>`_ research group in  `KU Leuven <https://www.kuleuven.be/english/>`_ through the senior research fellowship of Prof. Haroldo in 2018-2019, `CNPq <https://en.wikipedia.org/wiki/National_Council_for_Scientific_and_Technological_Development>`_ "Produtividade em Pesquisa" grant, `FAPEMIG <https://fapemig.br>`_ and the `GOAL <http://goal.ufop.br>`_ research group in the `Computing Department <http://www.decom.ufop.br>`_ of `UFOP <https://www.ufop.br/>`_.
