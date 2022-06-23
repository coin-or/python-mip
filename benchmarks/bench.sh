#!/bin/bash

rm -f queens-mip-cbc-cpython.csv queens-mip-cbc-pypy.csv queens-mip-grb-cpython.csv \
    queens-mip-grb-pypy.csv

echo "Benchmarking n-Queens CBC Python-MIP CPYTHON"
python3 queens.py cbc
mv queens-mip-cbc.csv queens-mip-cbc-cpython.csv

echo "Benchmarking n-Queens Gurobi Python-MIP CPYTHON"
python3 queens.py gurobi
mv queens-mip-gurobi.csv queens-mip-grb-cpython.csv

echo "Benchmarking n-Queens CBC Python-MIP Pypy"
pypy3 queens.py cbc
mv queens-mip-cbc.csv queens-mip-cbc-pypy.csv

echo "Benchmarking n-Queens Gurobi Python-MIP Pypy"
pypy3 queens.py gurobi
mv queens-mip-gurobi.csv queens-mip-grb-pypy.csv

echo "Benchmarking n-Queens Gurobi"
python3 queens-gurobi.py

echo "Benchmarking n-Queens Pulp CPYTHON"
python3 queens-pulp.py
mv queens-pulp.csv queens-pulp-cpython.csv

echo "Benchmarking n-Queens Pulp Pypy"
pypy3 queens-pulp.py
mv queens-pulp.csv queens-pulp-pypy.csv

rm -f rcpsp-mip-cbc-cpython.csv rcpsp-mip-grb-cpython.csv rcpsp-mip-cbc-pypy.csv rcpsp-mip-grb-pypy.csv
for solver in cbc gurobi;
do
    echo "Benchmarking RCPSP $solver"
    python3 rcpsp.py ./data/rcpsp/j301_1.sm $solver 20 >> rcpsp-mip-$solver-cpython.csv 
    pypy3 rcpsp.py ./data/rcpsp/j301_1.sm $solver 20  >> rcpsp-mip-$solver-pypy.csv
    python3 rcpsp.py ./data/rcpsp/j601_1.sm $solver 2 >> rcpsp-mip-$solver-cpython.csv
    pypy3 rcpsp.py ./data/rcpsp/j601_1.sm $solver 2 >> rcpsp-mip-$solver-pypy.csv
    python3 rcpsp.py ./data/rcpsp/j1201_1.sm $solver 1 >> rcpsp-mip-$solver-cpython.csv
    pypy3 rcpsp.py ./data/rcpsp/j1201_1.sm $solver 1 >> rcpsp-mip-$solver-pypy.csv
done
mv rcpsp-mip-gurobi-cpython.csv rcpsp-mip-grb-cpython.csv
mv rcpsp-mip-gurobi-pypy.csv rcpsp-mip-grb-pypy.csv


rm -f bmcp-mip-grb-cpython.csv bmcp-mip-grb-pypy.csv bmcp-mip-cbc-cpython.csv bmcp-mip-cbc-pypy.csv
for inst in ./data/bmcp/*.col;
do
    for solver in cbc gurobi;
    do
        echo "Benchmarking BMCP instance $inst with solver $solver and cpython"
        python3 bmcp.py $inst $solver | grep RRR | cut -d ':' -f 2 >> bmcp-mip-$solver-cpython.csv
        echo "Benchmarking BMCP instance $inst with solver $solver and pypy"
        pypy3 bmcp.py $inst $solver | grep RRR | cut -d ':' -f 2 >> bmcp-mip-$solver-pypy.csv
    done
done
mv bmcp-mip-gurobi-pypy.csv mv bmcp-mip-grb-pypy.csv
mv bmcp-mip-gurobi-cpython.csv mv bmcp-mip-grb-cpython.csv


pypy3 summarize.py

txt2tags -t tex bench-results.t2t
txt2tags -t rst bench-results.t2t

