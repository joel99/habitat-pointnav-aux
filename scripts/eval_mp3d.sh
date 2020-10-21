#!/bin/bash
hostname
if [[ $# -eq 2 ]]
then
    python -u habitat_baselines/run.py --run-type eval --exp-config habitat_baselines/config/mp3d_pn/$1.pn.yaml --ckpt-path ~/share/mp3d_pn/$1/$1.$2.pth
elif [[ $# -eq 4 ]]
then
    echo "Evaluating ${@:2}"
    for i in $(seq ${@:2})
    do
        echo $i
        python -u habitat_baselines/run.py --run-type eval --exp-config habitat_baselines/config/mp3d_pn/$1.pn.yaml --ckpt-path ~/share/mp3d_pn/$1/$1.$i.pth
    done
elif [[ $# -eq 5 ]]
then
    echo "Evaluating ${@:3}"
    for i in $(seq ${@:3})
    do
        echo $i
        python -u habitat_baselines/run.py --run-type eval --exp-config habitat_baselines/config/mp3d_pn/$1.pn.yaml --run-id $2 --ckpt-path ~/share/mp3d_pn/$1/$1.$i.pth
    done
else
    echo "Expected 2, 4, or 5 arguments to specify config. Format: variant + run id + 3 args to `seq`"
fi