#!/bin/bash
#SBATCH --job-name=gc
#SBATCH --gres gpu:1
hostname

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}


if [[ $# -lt 2 ]]
then
    echo "Expect two arguments to specify config file and run id"
elif [[ $# -eq 2 ]]
then
    python -u habitat_baselines/run.py --run-type train --exp-config habitat_baselines/config/gibson_r3/$1.pn.yaml --run-id $2
elif [[ $# -eq 3 ]]
then
    python -u habitat_baselines/run.py --run-type train --exp-config habitat_baselines/config/gibson_r3/$1.pn.yaml --run-id $2 --ckpt-path ~/share/golden/$1/$1.$3.pth
else
    for i in $(seq ${@:3})
    do
        echo $i
        python -u habitat_baselines/run.py --run-type eval --exp-config habitat_baselines/config/gibson_r3/$1.pn.yaml --run-id $2 --ckpt-path ~/share/golden/$1/$1.$i.pth
    done
fi