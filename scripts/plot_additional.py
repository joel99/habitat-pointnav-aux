"""
One-off script for getting loss curves
"""
#%%
import numpy as np
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import math
import os
import os.path as osp
import matplotlib.pyplot as plt
import glob
from scipy import interpolate
#%%
run_root = "/nethome/jye72/projects/embodied-recall/tb/official_gc/"
run_count = 3
# nested by variant and then run i
simple_paths = ['baseline', 'fpc1', 'fpc2', 'fpc4', 'fpc8', 'fpc16', 'inverse', 'frames']

five_paths = ['fpc_attn', 'fpc_attn-e', 'fpc_fixed', 'fpc_single', 'fpc_repeat', 'fpc16w']
seven_paths = [ 'fpcit_attn', 'fpcit_attn-e',
    'fpcit_single', 'fpcit_uniform', 'fpcit_soft']
that_guy = ['fpci_attn-e']
prefixes = ['simple', 'five_tasks', 'seven_tasks', '.']
run_groups = [simple_paths, five_paths, seven_paths, that_guy]
run_paths = {}
for prefix, group in zip(prefixes, run_groups):
    for path in group:
        run_paths[path] = f"{prefix}/{path}"

def get_run_logs(v):
    folder = os.path.join(run_root, run_paths[v])
    run_folders = os.listdir(folder)
    event_paths = []
    for i, run_folder in enumerate(run_folders):
        if i >= run_count:
            print(f"exiting {v}")
            break
        full_path = os.path.join(folder, run_folder)
        event_paths.append(full_path)
    return event_paths

#%%
variants = ['fpcit_attn-e'] # ['baseline'] #, 'fpc8', 'inverse', 'frames', 'fpcit_attn-e', 'fpcit_single', 'fpc_single', 'fpc16w', 'fpcit_soft', 'fpcit_uniform']

variant_paths = {}
for variant in variants:
    variant_paths[variant] = get_run_logs(variant)

#%%
# Get loss scalars
cpca = "CPC|A-"
tasks = {
    'cpctask_full': f"{cpca}1",
    'fpctask_a': f"{cpca}2",
    'fpctask_b': f"{cpca}4",
    'fpctask_c': f"{cpca}8",
    'fpctask_d': f"{cpca}16",
    'inversedynamicstask': f"ID",
    'temporaldistancetask': f"TD",
}

# Loading too much data is slow...
tf_size_guidance = {
    'scalars': 1000,
}

# check - if losses in folder, use that

plot_values = {}
plot_steps = {}
for variant, variant_runs in variant_paths.items():
    min_steps = 0
    for i, run in enumerate(variant_runs):
        accum_path = os.path.join(run, plot_key_folder)
        loss_data = {}
        if osp.exists(osp.join(accum_path, 'losses')):
            for task in tasks:
                loss_path = osp.join(accum_path, 'losses', task)
                event_acc = EventAccumulator(loss_path, tf_size_guidance)
                event_acc.Reload()
                print(event_acc.Tags())
                scalars = event_acc.Scalars('losses')
                steps_and_values = np.stack(
                    [np.asarray([scalar.step, scalar.value])
                    for scalar in scalars])
                # # unload steps to make sure the number of frames is equal across runs
                steps = steps_and_values[:, 0]
                values = steps_and_values[:, 1]
                plot_steps[task] = [steps]
                plot_values[task] = [values]
        else:
            print("Nope")

# %%
# do a binary search for the closest step to 40.96 M, trim.

TARGET_STEP = 40.96e6
HEURISTIC = .72 # baseline is the longest (I think and it's around .6)
def get_crop_index(steps):
    index = int(len(steps) * HEURISTIC)
    while steps[index] <= TARGET_STEP:
        index += 1
    return index
# actually, let's just do a linear search

cropped_steps = {}
cropped_values = {}
desired_steps = np.arange(818) * .5e5 # we'll have about 75 -> .5M steps. Let's interpolate 80 steps
interpolated_values = {}
for variant in tasks:
    index = get_crop_index(plot_steps[variant][0])
    cropped_steps[variant] = []
    cropped_values[variant] = []
    interpolated_values[variant] = []
    for run in range(run_count):
        steps = plot_steps[variant][run][:index]
        values = plot_values[variant][run][:index]
        cropped_steps[variant].append(steps)
        cropped_values[variant].append(values)
        interpolated = np.interp(desired_steps, steps, values)
        interpolated_values[variant].append(interpolated)

# plt.plot(cropped_steps[var][0], cropped_values[var][0])
# plt.plot(desired_steps, interpolated_values[var][0])



#%%
# a special just for fpc16 since the run is messed up/multiple runs
messed_up_key = "fpci_attn-e"
messed_up_key = "fpc16"
messed_up_key = "baseline"
if messed_up_key in plot_steps:
    for i in range(run_count):
        unique, indices = np.unique(plot_steps[messed_up_key][i], return_index=True)
        print(unique)
        # indices = indices[unique < 82]
        plot_steps[messed_up_key][i] = plot_steps[messed_up_key][i][indices]#[sorted_indices]
        plot_values[messed_up_key][i] = plot_values[messed_up_key][i][indices]#[sorted_indices]

#%%
WINDOW_SIZE = 3
plot_means = {}
plot_stds = {}
plot_ci = {}
for variant in plot_values:
    if 'eval' in plot_key:
        data = np.array(plot_values[variant])
    else:
        data = np.array(interpolated_values[variant])
    values_smoothed = np.empty_like(data)
    if WINDOW_SIZE > 1:
        for i in range(data.shape[1]):
            window_start = max(0, i - WINDOW_SIZE)
            window = data[:, window_start:i + 1]
            values_smoothed[:, i] = window.mean(axis=1)
    else:
        values_smoothed = data
    plot_means[variant] = np.mean(values_smoothed, axis=0)
    plot_stds[variant] = np.std(values_smoothed, axis=0)
    plot_ci[variant] = 1.96 * plot_stds[variant] / math.sqrt(run_count) # 95%

#%%
key_labels = {  "spl": "SPL - Train",
                "success": "Success - Train",
                "eval_spl": "SPL - Val",
                "eval_success": "Success - Val"
            }
axis_labels = {
    "spl": "SPL",
    "eval_spl": "SPL",
    "success": "Success",
    "eval_success": "Success"
}

cpc_name = "CPC|A"
cpc_all_name = cpc_name + "{1-16}"
variant_labels = {
    "baseline": "Baseline",
    "fpc1": f"{cpc_name}-1",
    "fpc2": f"{cpc_name}-2",
    "fpc4": f"{cpc_name}-4",
    "fpc8": f"{cpc_name}-8",
    "fpc16": f"{cpc_name}-16",
    "inverse": "ID",
    "frames": "TD",
    "fpc16w": f"Weighted {cpc_name}",
    "fpc_attn": f"{cpc_all_name}: Attn", #
    "fpc_attn-e": f"{cpc_all_name}: Attn+E",
    "fpc_repeat": "CPC|A-16 Repeat",
    "fpc_fixed": f"{cpc_all_name}: Fixed",
    "fpc_single": f"{cpc_all_name}: Single", # for use in last plot
    "fpcit_single": f"{cpc_all_name}+ID+TD: Single",
    "fpcit_uniform": f"{cpc_all_name}+ID+TD: Average",
    "fpcit_soft": f"{cpc_all_name}+ID+TD: Softmax",
    "fpcit_attn-e": f"{cpc_all_name}+ID+TD: Attn+E",
    "fpcit_attn": f"{cpc_all_name}: Attn",
    "fpci_attn-e": f"{cpc_all_name}+ID: Attn+E"
}
SMALL_SIZE = 12
MEDIUM_SIZE = 15
LARGE_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
x_scale = 2 if 'eval' in plot_key else 1e6
plt.style.use('seaborn-muted')
plt.figure(figsize=(6,4))
plt.xlabel("Frames (Million)")
plt.ylabel("Aux Task Loss")
plt.xticks(np.arange(0, 4.1e7, 1e7))

if 'eval' in plot_key and False:
    plt.xticks(np.arange(5, 45, 5))
    lower_lim = .5 if 'success' in plot_key else .3
    upper_lim = .9 if 'success' in plot_key else .75
    plt.ylim(lower_lim, upper_lim)
    plt.yticks(np.arange(lower_lim, upper_lim + 0.1, 0.1))
    plt.xlim(5, 40)
# plt.title(f"{key_labels[plot_key]}", fontsize=LARGE_SIZE)
spine_alpha = 0.3
plt.gca().spines['right'].set_alpha(spine_alpha)
plt.gca().spines['bottom'].set_alpha(spine_alpha)
plt.gca().spines['left'].set_alpha(spine_alpha)
plt.gca().spines['top'].set_alpha(spine_alpha)

plt.grid(alpha=0.25)
plt.tight_layout()

sns.reset_orig()  # get default matplotlib styles back
clrs = sns.color_palette('hls', n_colors=len(tasks))  # a list of RGB tuples
variant_labels = tasks
for i, variant in enumerate(tasks):
    if 'eval' in plot_key:
        x = plot_steps[variant][0] / x_scale
    else:
        x = desired_steps
    y = plot_means[variant]
    line, = plt.plot(x, y, label=variant_labels.get(variant, variant))
    line.set_color(clrs[i])
    plt.fill_between(x, y - plot_ci[variant], y + plot_ci[variant], facecolor=line.get_color(), alpha=0.5)
leg = plt.legend(loc=(1.04, 0.2), frameon=True)
# leg = plt.legend(loc=(.32, 0.02), ncol=1, frameon=True) # .72 for one col
# leg = plt.legend(loc=(.36, 0.02), ncol=1, frameon=True) # .72 for one col
# leg = plt.legend(loc=(.71, 0.02), ncol=1, frameon=True) # .72 for one col

for line in leg.get_lines():
    line.set_linewidth(2.0)
plt.savefig('test.pdf',bbox_inches="tight")
plt.show()

#%%
print(plot_key)
ten_mil_key = 5 # 10 12 14 16 18 20
if 'eval' not in plot_key:
    ten_mil_key = int(len(desired_steps) / 4) # because interpolated is going to have this length
    print(desired_steps[ten_mil_key])
for variant in variants:
    print(f"Variant: {variant:8} \t {plot_key} 10M: {plot_means[variant][ten_mil_key]:.3f} + {plot_ci[variant][ten_mil_key]:.3f} \t 40M: {plot_means[variant][-1]:.3f} + {plot_ci[variant][-1]:.3f}")