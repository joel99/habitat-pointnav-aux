#!/bin/bash
#SBATCH --job-name=quick_eval_gc
#SBATCH --gres gpu:1
#SBATCH -p short

hostname
# Detailed runs i.e. collect diagnostics
if [[ $# -lt 1 ]]
then
    echo "Expect one argument to specify config file"
elif [[ $# -eq 2 ]] # <variant> <ckpt>
then
    python -u habitat_baselines/run_detailed.py --run-type eval --exp-config habitat_baselines/config/pointnav_gc/$1.pn.yaml --ckpt-path ~/share/pn_gc_ckpts/$1/$1.$2.pth
elif [[ $# -eq 3 ]] # <variant> <ckpt> <run id>
then
    python -u habitat_baselines/run_detailed.py --run-type eval --exp-config habitat_baselines/config/official_gc/$1.pn.yaml --ckpt-path ~/share/r2/$1/run_$3.$2.pth --run-id $3
else
    echo "Arguments misconfigured"
fi