
import numpy as np
import pandas as pd
import os
import os.path as osp
import json
import glob
import math
import json
cpc = 'cpca'
allcpc = 'cpca-id-td'
eval_stat_root = '/nethome/jye72/share/r3_detailed/'
with open('./scripts/eval_success_ckpts.csv', 'r') as f:
    success_dict = json.load(f)
with open('./scripts/eval_spl_ckpts.csv', 'r') as f:
    spl_dict = json.load(f)
params = []
for variant in success_dict:
    for run in range(len(success_dict[variant])):
        ckpt_success = success_dict[variant][run]
        ckpt_spl = spl_dict[variant][run]
        succ_fn = osp.join(eval_stat_root, f"{variant}_{ckpt_success}_run_{run}.json")
        spl_fn = osp.join(eval_stat_root, f"{variant}_{ckpt_spl}_run_{run}.json")
        if not osp.exists(succ_fn):
            params.append((variant, ckpt_success, run))
        if ckpt_success != ckpt_spl and not osp.exists(spl_fn):
            params.append((variant, ckpt_spl, run))

# launch them
for param in params:
    variant, ckpt, run = param
    os.system(f'sbatch -x calculon ./scripts/run_gc_detailed.sh {variant} {ckpt} {run}')