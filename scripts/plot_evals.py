"""
Using this eval script
- modify cell 2 definitions as desired (load in the appropriate folders)
- get values in last cell, plots in second to last cell
- modify plot key to see given metric
"""
#%%
import math
import os
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Strings
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
    # root_folder = os.path.join(run_root, run_paths[v])
    # if len(os.listdir(root_folder)) < run_count:
    #     print(f"{root_folder} has insufficient runs")
    #     return
    # folders = [os.path.join(root_folder, run_folder) for run_folder in os.listdir(root_folder)]
    folder = os.path.join(run_root, run_paths[v])
    run_folders = os.listdir(folder)
    event_paths = []
    i = 0
    for run_folder in run_folders:
        if int(run_folder.split('_')[-1]) > 3:
            continue
        if i >= run_count:
            print(f"exiting {v}")
            break
        i += 1
        full_path = os.path.join(folder, run_folder)
        event_paths.append(full_path)
    return event_paths

#%%
# Set what to plot
variants_1 = ['baseline', 'fpc8', 'inverse', 'frames']
variants_2 = ['baseline', 'fpc8', 'fpc16w', 'fpc_single', 'fpcit_single']
variants_3 = ['baseline', 'fpcit_attn-e', 'fpcit_uniform', 'fpcit_soft', 'fpcit_single']
plotted_union = list(set(variants_1) | set(variants_2) | set(variants_3))
palette = sns.color_palette(palette='muted', n_colors=len(plotted_union), desat=0.9)

variant_colors = {}
for i, v in enumerate(plotted_union):
    variant_colors[v] = palette[i]
sns.palplot(palette)

# plot_key = 'success' # spl, success, eval_spl, eval_success
# plot_key = 'spl' # spl, success, eval_spl, eval_success
plot_key = 'eval_success' # spl, success, eval_spl, eval_success
# plot_key = 'eval_spl' # spl, success, eval_spl, eval_success
variants = ['baseline'] # plotted_union

variant_paths = {}
for variant in variants:
    variant_paths[variant] = get_run_logs(variant)
plot_key_folder_dict = {
    'eval_spl': 'eval_spl_average spl/',
    'eval_success': 'eval_success_average success/'
}
plot_key_folder = plot_key_folder_dict.get(plot_key, "")
tf_size_guidance = {'scalars': 1000}


#%%

# Train val curves hack
plot_keys = ["spl", "eval_spl", "eval_success", "success"]
for key in plot_keys:
    plot_values[key] = []
    plot_steps[key] = []

# We want the runs, but instead of multiple variants we use multiple keys
print(variant_paths)
paths = variant_paths['baseline']
variant_paths = {
    key: paths for key in plot_keys
}

variants = plot_keys
print(plot_steps)
# print(plot_steps['spl'][])

#%%
# Load
plot_values = {}
plot_steps = {}
for variant, variant_runs in variant_paths.items():
    plot_values[variant] = []
    plot_steps[variant] = []
    min_steps = 0
    # plot_key_folder = plot_key_folder_dict.get(variant, "")
    for i, run in enumerate(variant_runs):
        accum_path = os.path.join(run, plot_key_folder)
        event_acc = EventAccumulator(accum_path, tf_size_guidance)
        event_acc.Reload()
        scalars = event_acc.Scalars(plot_key)
        # scalars = event_acc.Scalars(variant)
        steps_and_values = np.stack(
            [np.asarray([scalar.step, scalar.value])
            for scalar in scalars])
        steps = steps_and_values[:, 0]
        values = steps_and_values[:, 1]
        plot_steps[variant].append(steps)
        plot_values[variant].append(values)

# %%
# Train plot
# do a binary search for the closest step to 40.96 M, trim.
TARGET_STEP = 40.96e6
HEURISTIC = .2 # baseline is the longest (I think and it's around .6)
def get_crop_index(steps):
    index = int(len(steps) * HEURISTIC)
    while steps[index] <= TARGET_STEP:
        index += 1
    return index
# actually, let's just do a linear search

interpolated_values = {}
# for key in plot_keys:
#     if 'eval' not in key:
if 'eval' not in plot_key:
    cropped_steps = {}
    cropped_values = {}
    desired_steps = np.arange(0, 818, 4) * .5e5 # we'll have about 75 -> .5M steps. Let's interpolate 80 steps
    interpolated_values = {}
    for variant in variants:
    # variant = key
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

print(interpolated_values.keys())

#%%
# Train val cell
# Fix baseline numbers
print(key)
for key in plot_steps:
    if 'eval' in key:
        data = np.array(plot_values[variant])
    else:
        data = np.array(interpolated_values[variant])
    messed_up_key = key
    for i in range(run_count):
        unique, indices = np.unique(plot_steps[messed_up_key][i], return_index=True)
        if 'eval' in key:
            indices = indices[unique < 82]
            plot_steps[messed_up_key][i] = plot_steps[messed_up_key][i][indices]#[sorted_indices]
            plot_values[messed_up_key][i] = plot_values[messed_up_key][i][indices]#[sorted_indices]
        else:
            indices = indices[unique > 5e6]
            plot_steps[messed_up_key][i] = plot_steps[messed_up_key][i][indices]#[sorted_indices]
            plot_values[messed_up_key][i] = plot_values[messed_up_key][i][indices]#[sorted_indices]

#%%
# Cropping variants to the appropriate checkpoints
for messed_up_key in variants:
    if messed_up_key in plot_steps:
        for i in range(run_count):
            unique, indices = np.unique(plot_steps[messed_up_key][i], return_index=True)
            indices = indices[unique < 82]
            plot_steps[messed_up_key][i] = plot_steps[messed_up_key][i][indices]#[sorted_indices]
            plot_values[messed_up_key][i] = plot_values[messed_up_key][i][indices]#[sorted_indices]


def get_means_and_ci(values, window_size=1):
    r"""
        Returns means and CI np arrays
        args:
            values: dict of trials by variant, each value a list of trial data
            window_size: window smoothing of trials
        returns:
            mean and CI dict, keyed by same variants
    """
    means={}
    ci = {}
    for variant in values:
        data = np.array(values[variant])
        values_smoothed = np.empty_like(data)
        if window_size > 1:
            for i in range(data.shape[1]):
                window_start = max(0, i - window_size)
                window = data[:, window_start:i + 1]
                values_smoothed[:, i] = window.mean(axis=1)
        else:
            values_smoothed = data

        best_until = np.copy(values_smoothed)
        for t in range(best_until.shape[1]):
            best_until[:,t] = np.max(best_until[:,:t+1], axis=1)
        values_smoothed = best_until

        means[variant] = np.mean(values_smoothed, axis=0)
        ci[variant] = 1.96 * np.std(values_smoothed, axis=0) \
            / math.sqrt(run_count) # 95%

    return means, ci

if 'eval' in plot_key:
    data = plot_values
else:
    data = interpolated_values
plot_means, plot_ci = get_means_and_ci(data, window_size=1) # 3

#%%
# Plot train val
variant_labels = {
    "spl": "Train",
    "success": "Train",
    "eval_spl": "Val",
    "eval_success": "Val"
}
for variant in variants:
    x_scale = 2 if 'eval' in variant else 1e6
    if 'eval' in variant:
        x = plot_steps[variant][0] / x_scale
    else:
        x = desired_steps / x_scale
    y = plot_means[variant]
    line, = plt.plot(x, y, label=variant_labels.get(variant, variant))
    plt.fill_between(x, y - plot_ci[variant], y + plot_ci[variant], facecolor=line.get_color(), alpha=0.5)
leg = plt.legend(loc=(.78, 0.02), ncol=1, frameon=True) # .72 for one col
for line in leg.get_lines():
    line.set_linewidth(2.0)
plt.savefig('test2.pdf', dpi=150, bbox_inches="tight")

#%%
# Style
SMALL_SIZE = 12
MEDIUM_SIZE = 15
LARGE_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.style.use('seaborn-muted')
plt.figure(figsize=(6,4))

plt.xlabel("Frames (Million)")
plt.ylabel(key_labels[plot_key])
spine_alpha = 0.3
plt.gca().spines['right'].set_alpha(spine_alpha)
plt.gca().spines['bottom'].set_alpha(spine_alpha)
plt.gca().spines['left'].set_alpha(spine_alpha)
plt.gca().spines['top'].set_alpha(spine_alpha)
plt.grid(alpha=0.25)
plt.tight_layout()

# Plot evals
# Axes
plt.xlim(5, 40)
plt.xticks(np.arange(5, 45, 5))

if 'eval' in plot_key:
    lower_lim = .4 if 'success' in plot_key else .3
    upper_lim = .9 if 'success' in plot_key else .75

    plt.ylim(lower_lim, upper_lim)
    plt.yticks(np.arange(lower_lim, upper_lim + 0.01, 0.1))

x_scale = 2 if 'eval' in plot_key else 1e6
for variant in ['baseline']:
    if 'eval' in plot_key:
        x = plot_steps[variant][0] / x_scale
    else:
        x = desired_steps
    y = plot_means[variant]
    line, = plt.plot(x, y, label=variant_labels.get(variant, variant), c=variant_colors.get(variant))
    plt.fill_between(x, y - plot_ci[variant], y + plot_ci[variant], facecolor=line.get_color(), alpha=0.5)

def annotate(index, from_var, to_var, hoffset=-6, voffset=0):
    lo = plot_means[from_var][index-5]
    hi = plot_means[to_var][index-5]
    plt.annotate("", xy=(index, lo), xycoords="data", xytext=(index, hi), textcoords="data", arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0"))
    plt.text(index+hoffset, hi+voffset, f"+{(hi - lo):.2f}",size=16)

# Simple
# annotate(35, "baseline", "fpc8")
# annotate(15, "baseline", "fpc8")
leg_start = .71

# Homo
# annotate(35, "baseline", "fpc8", -6, -0.05)
# annotate(36, "baseline", "fpcit_single", -6, 0.02)
# leg_start = .36

# Diverse
# leg_start = .32
# annotate(36, "baseline", "fpcit_single", 0.5, -0.02)
# annotate(35, "baseline", "fpcit_attn-e", -6)
# annotate(10, "fpcit_single", "fpcit_attn-e", 0.5, -0.1)

leg = plt.legend(loc=(leg_start, .01),
    markerfirst=False, ncol=1, frameon=False, labelspacing=0.4)
for line in leg.get_lines():
    line.set_linewidth(2.0)

plt.savefig('test.pdf', dpi=150)
#%%
# Prints values for table
print(plot_key)
ten_mil_key = 5 # 10 12 14 16 18 20
if 'eval' not in plot_key:
    ten_mil_key = int(len(desired_steps) / 4) # because interpolated is going to have this length
    print(desired_steps[ten_mil_key])
for variant in variants:
    print(f"Variant: {variant:8} \t {plot_key} 10M: {plot_means[variant][ten_mil_key]:.3f} + {plot_ci[variant][ten_mil_key]:.3f} \t 40M: {plot_means[variant][-1]:.3f} + {plot_ci[variant][-1]:.3f}")