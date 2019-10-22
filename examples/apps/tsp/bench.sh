#! /bin/sh
#
# bench.sh
# Copyright (C) 2019 haroldo <haroldo@soyuz>
#
# Distributed under terms of the MIT license.
#

rm -f results.csv
rm -f runbench.sh
idir='/home/haroldo/inst/tsp/'
insts='ulysses16 ulysses22 att48 bier127 gr202 lin318 d493'
TIMELIMIT=36000


for inst in ${insts};
do
    rm -f trunbench-${inst}.sh
done

for inst in ${insts};
do
    ifile=${idir}/${inst}.tsp
    for solver in GUROBI CBC;
    do
        for cut in 0 1;
        do
            for lazy in 0 1;
            do
                for heur in 0 1;
                do
                    export SOLVER_NAME=$solver
                    flog=${inst}-${solver}-${cut}-${lazy}-${heur}.log
                    ferror=${inst}-${solver}-${cut}-${lazy}-${heur}.error
                    echo "export SOLVER_NAME=${solver} ; pypy3 tsp.py $ifile ${TIMELIMIT} 1 ${cut} ${lazy} ${heur} > $flog 2> $ferror" >> trunbench-${inst}.sh
               done
            done
        done
    done
done

for inst in ${insts};
do
    cat trunbench-${inst}.sh | sort -R > rb-${inst}.sh
    rm -f  trunbench-${inst}.sh
    chmod u+x  rb-${inst}.sh
done


