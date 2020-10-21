#!/bin/bash
#SBATCH --job-name=gc
#SBATCH --gres gpu:1
hostname
if [[ $# -lt 1 ]]
then
    echo "Expect one argument to specify config file"
elif [[ $# -eq 1 ]]
then
    python -u habitat_baselines/run.py --run-type train --exp-config habitat_baselines/config/official_gc/$1.pn.yaml
elif [[ $# -eq 2 ]]
then
    python -u habitat_baselines/run.py --run-type train --exp-config habitat_baselines/config/official_gc/$1.pn.yaml --ckpt-path ~/share/golden/$1/$1.$2.pth
else
    for i in $(seq ${@:2})
    do
        echo $i
        python -u habitat_baselines/run.py --run-type eval --exp-config habitat_baselines/config/pointnav_gc/$1.pn.yaml --ckpt-path ~/share/golden/$1/$1.$i.pth
    done
fi