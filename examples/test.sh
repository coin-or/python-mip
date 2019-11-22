#! /bin/sh

for solver in cbc gurobi;
do
    for file in ./knapsack.py ./jssp.py ./tsp-compact.py queens.py bmcp.py rcpsp.py;
    do
        export SOLVER_NAME="$solver"
        pypy3 $file
    done
done

