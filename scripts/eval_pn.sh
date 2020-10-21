#!/bin/bash
hostname
if [[ $# -eq 2 ]]
then
    python -u habitat_baselines/run.py --run-type eval --exp-config habitat_baselines/config/gibson_r3/$1.pn.yaml --ckpt-path ~/share/r3/$1/$1.$2.pth
elif [[ $# -eq 4 ]]
then
    echo "Evaluating ${@:2}"
    for i in $(seq ${@:2})
    do
        echo $i
        python -u habitat_baselines/run.py --run-type eval --exp-config habitat_baselines/config/gibson_r3/$1.pn.yaml --ckpt-path ~/share/r3/$1/$1.$i.pth
    done
elif [[ $# -eq 5 ]]
then
    all_ckpts=""
    for i in $(seq ${@:3})
    do
        all_ckpts+="/nethome/jye72/share/r3/$1/$1.$i.pth,"
    done
    python -u habitat_baselines/run.py --run-type eval --exp-config habitat_baselines/config/gibson_r3/$1.pn.yaml --run-id $2 --ckpt-path ${all_ckpts}
    # echo "Evaluating ${@:3}"
    # for i in $(seq ${@:3})
    # do
    #     echo $i
    #     python -u habitat_baselines/run.py --run-type eval --exp-config habitat_baselines/config/gibson_r3/$1.pn.yaml --run-id $2 --ckpt-path ~/share/r3/$1/$1.$i.pth
    # done
else
    echo "Expected 2, 4, or 5 arguments to specify config. Format: variant + run id + 3 args to `seq`"
fi