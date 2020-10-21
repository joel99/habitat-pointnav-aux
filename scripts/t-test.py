#%%
import numpy as np
import pandas as pd
import os
import os.path as osp
import json
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import PIL.Image
import glob
import math
cpc = 'cpca'
allcpc = 'cpca-id-td'
eval_stat_root = '/nethome/jye72/share/r3_detailed/'

def add_info(data, variant, run, entry_list):
    """ data: dict
        entry_list: list to append to
    """
    num_eps = len(data)
    for i, ep in enumerate(data):
        stats = ep['stats']
        episode_info = ep['episode_info']
        reward, _, success, spl = stats.values()
        ep_id, scene_id, start_pos, start_rot, more_ep_info, _, goals, start_room, shortest_path = episode_info.values()
        entry_list.append({
            "index": i,
            "ep": ep_id,
            "scene": scene_id,
            "spl": spl,
            "success": success,
            "variant": variant,
            "run": int(run)
        })

#%%

# Get me a list I need to run
import json
with open('eval_success_ckpts.csv', 'r') as f:
    success_dict = json.load(f)
with open('eval_spl_ckpts.csv', 'r') as f:
    spl_dict = json.load(f)
print("What you need to run in detailed mode")
all_runs = []
for variant in success_dict:
    for run in range(len(success_dict[variant])):
        ckpt_success = success_dict[variant][run]
        ckpt_spl = spl_dict[variant][run]
        succ_fn = osp.join(eval_stat_root, f"{variant}_{ckpt_success}_run_{run}.json")
        spl_fn = osp.join(eval_stat_root, f"{variant}_{ckpt_spl}_run_{run}.json")
        if not osp.exists(succ_fn):
            print(f"{variant}, {ckpt_success}, {run}")
        if ckpt_success != ckpt_spl and not osp.exists(spl_fn):
            print(f"{variant}, {ckpt_spl}, {run}")


#%%
source_dict = success_dict
entries = []

# for run_stat_path in os.listdir(eval_stat_root):
for variant in source_dict:
    for run_id, ckpt in enumerate(source_dict[variant]):
        eval_fn = f"{variant}_{ckpt}_run_{run_id}.json"
        log_path = osp.join(eval_stat_root, eval_fn)
        # variant_dir = glob.glob(osp.join(eval_stat_root, variant, "*"))# os.listdir(osp.join(eval_stat_root, variant))
        # variant, ckpt, _, run_id = run_stat_path.split(".")[0].split("_")
        with open(log_path, 'r') as f:
            data = json.load(f)
            add_info(data, variant, run_id, entries)

# print(entries)
#%%
df = pd.DataFrame(entries, columns=["ep", "scene", "spl", "success", "variant", "run"])
for col in ["spl", "success", "ep", "run"]:
    df[col] = pd.to_numeric(df[col])

df.head()

#%%
print(df[df['variant'] == f"{allcpc}_soft"]["spl"].mean())
print(df[df['variant'] == f"{allcpc}_attn"]["spl"].mean())
print(df[df['variant'] == f"{allcpc}_attn-e"]["spl"].mean())
#%%
check_key = "spl"

variants = df['variant'].unique()
pair_results = {}
for variant in variants:
    pair_results[variant] = {}
# df['variant'] = pd.Categorical(df['variant'], variants_order)
for pair_1 in variants:
    for pair_2 in variants: # variants:
        if pair_1 == pair_2:
            continue
        pair = [pair_1, pair_2]
        pair_df = df[df['variant'].isin(pair)][["variant", "ep", check_key, "run"]]
        # sort it so that pair_1 is less than pair_2 so we diff pair_1 - pair_2
        # null hyp: pair_1 <= pair_2
        # alt hyp: pair_1 > pair_2
        # ! Ok, I don't know why the sign is flipped, print statement suggests opposite
        pair_df = pair_df.sort_values(by='variant', ascending=pair[0] > pair[1]) # ! should actualy be < according to print
        # print(pair_df.head())
        diff = pair_df.groupby(["ep", "run"])[check_key].diff()
        mean = diff.mean()
        n = diff.count()
        std = diff.std()
        t = (mean * math.sqrt(n) / std)
        p = 1 - stats.t.cdf(t, df=n - 1)
        pair_results[pair_1][pair_2] = (t, p, p < 0.05)

#%%
pair_1 = f"{cpc}4"
for pair_2 in pair_results[pair_1]:
    t, p, signif = pair_results[pair_1][pair_2]
    print(f"{pair_2:8} \t t - {t:.2f}, p - {p:.2f}, sig: {signif}")

#%%
pair_2 = f"{cpc}2"
for pair_1 in pair_results:
    if pair_1 == pair_2:
        continue
    t, p, signif = pair_results[pair_1][pair_2]
    if signif:
        print(f"{pair_1:8} \t t - {t:.2f}, p - {p:.2f}, sig: {signif}")
#%%
def get_pareto(var_list):
    # gets the top results of a list (like a table for ex)
    pareto = var_list.copy()
    for pair_1 in var_list:
        for pair_2 in var_list:
            if pair_1 == pair_2:
                continue
            t, p, signif = pair_results[pair_1][pair_2]
            if signif and pair_2 in pareto:
                # pair_1 is better than pair_2
                pareto.remove(pair_2)
    return pareto

simple = ['baseline', f"{cpc}1", f"{cpc}2", f"{cpc}4", f"{cpc}8", f"{cpc}16", "id", "td"]
homo = [f"{cpc}4", f"{cpc}16w", f"{cpc}_single", f"{allcpc}_single"]
ssdiverse = [f'{allcpc}_single', f'{cpc}4', f'{allcpc}_attn-2e', f'{allcpc}_average', f'{allcpc}_soft', f'{allcpc}_attn']
ablations = [f'{cpc}_single', f'{cpc}_attn', f'{cpc}_attn-e', f'{cpc}_fixed', f'{cpc}_repeat']
print(f"{check_key}")
print(f"Simple: {get_pareto(simple)}")
print(f"Homo: {get_pareto(homo)}")
print(f"Diverse: {get_pareto(diverse)}")
print(f"CPC: {get_pareto(ablations)}")
print(f"All: {get_pareto(list(variants))}")

#%%
"""
Notes:
success @ 40M
fpc_repeat
fpcit_attn

success @ 10M
fpcit_attn-e
fpci_attn-e
fpcit_attn
fpc_attn-e

spl @ 40M
fpcit_attn-e
fpc_repeat
fpcit_soft

spl@10M
fpcit_attn-e
fpcit_uniform
fpc_attn-e
"""

#%%
test_df = pd.DataFrame({
    'variant': ['baseline', 'baseline', 'baseline', 'baseline', 'baseline', 'baseline',\
        'fpc1', 'fpc1', 'fpc1', 'fpc1', 'fpc1', 'fpc1'],
    'ep': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
    'spl': [0.1, 0.2, 0.3, 0.3, 0.2, 0.4, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4],
    'run': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
})
# Ok, the sign doesn't flip here even though print flips
for pair_1 in ['fpc1']:
    for pair_2 in ['baseline']: # variants:
        if pair_1 == pair_2:
            continue
        pair = [pair_1, pair_2]
        pair_df = test_df[test_df['variant'].isin(pair)][["variant", "ep", "spl", "run"]]
        # sort it so that pair_1 is less than pair_2 so we diff pair_1 - pair_2
        # null hyp: pair_1 <= pair_2
        # alt hyp: pair_1 > pair_2
        pair_df = pair_df.sort_values(by='variant', ascending=pair[0] > pair[1])
        print(pair_df.head())
        diff = pair_df.groupby(["ep", "run"])["spl"].diff()
        mean = diff.mean()
        print(mean)

