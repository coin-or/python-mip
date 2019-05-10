#!/bin/bash

rm -f *.csv

python3 queens.py cbc
mv queens-mip-cbc.csv queens-mip-cbc-cpython.csv

python3 queens.py gurobi
mv queens-mip-gurobi.csv queens-mip-gurobi-cpython.csv

pypy3 queens.py cbc
mv queens-mip-cbc.csv queens-mip-cbc-pypy.csv

pypy3 queens.py gurobi
mv queens-mip-gurobi.csv queens-mip-gurobi-pypy.csv

python3 queens-gurobi.py

python3 queens-pulp.py
mv queens-pulp.csv queens-pulp-cpython.csv

pypy3 queens-pulp.py
mv queens-pulp.csv queens-pulp-pypy.csv

