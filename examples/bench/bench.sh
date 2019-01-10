#!/bin/bash

timeLimit=4000

for n in 50 100 150 200 250 300 350 400 450 500;
do
    utime=$(time -p python queens.py $n -solver=cbc -justmodel=1 > /dev/null 2>&1 | grep -i "user" | cut -d ' ' -f 2 ; )
    echo AAA $utime
    
done
