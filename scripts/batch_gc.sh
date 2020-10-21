#!/bin/bash
hostname
# sbatch format: sbatch --array=0-3%1 ./run_gc.sh <prefix>
echo "Starting run ${SLURM_ARRAY_TASK_ID}"
if [[ $# -lt 1 ]]
then
    echo "Expect one argument to specify config file"
elif [[ $# -eq 1 ]]
then
    echo "Starting training with run id ${SLURM_ARRAY_TASK_ID}"
    python -u habitat_baselines/run.py --run-type train --exp-config habitat_baselines/config/official_gc/$1.pn.yaml --run-id ${SLURM_ARRAY_TASK_ID}
elif [[ $# -eq 2 ]]
then
    echo "Starting training with run id $2"
    python -u habitat_baselines/run.py --run-type train --exp-config habitat_baselines/config/official_gc/$1.pn.yaml --run-id $2
    # python -u habitat_baselines/run.py --run-type train --exp-config habitat_baselines/config/official_gc/$1.pn.yaml --run-id $2
elif [[ $# -eq 3 ]] # This is for resuming training - pretend to be sbatch
then
    echo "Resuming training with run id $3"
    python -u habitat_baselines/run.py --run-type train --exp-config habitat_baselines/config/official_gc/$1.pn.yaml --run-id $3 --ckpt-path ~/share/r2/$1/$1.$2.pth
else
    echo "If you're looking to eval, use eval_gc_single"
fi