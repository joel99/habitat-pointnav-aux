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
from sklearn import metrics

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
cpc_codename = "cpca"
cpca_id_td_codename = "cpca-id-td"
cpc_all_name = cpc_name + "{1-16}"
variant_labels = {
    "baseline": "Baseline",
    f"{cpc_codename}1": f"{cpc_name}-1",
    f"{cpc_codename}2": f"{cpc_name}-2",
    f"{cpc_codename}4": f"{cpc_name}-4",
    f"{cpc_codename}8": f"{cpc_name}-8",
    f"{cpc_codename}16": f"{cpc_name}-16",
    "id": "ID",
    "td": "TD",
    f"{cpc_codename}16w": f"Weighted {cpc_name}",
    f"{cpc_codename}_attn": f"{cpc_all_name}: Attn",
    f"{cpc_codename}_attn-e": f"{cpc_all_name}: Attn+E",
    f"{cpc_codename}_repeat": "CPC|A-16 Repeat",
    f"{cpc_codename}_fixed": f"{cpc_all_name}: Fixed",
    f"{cpc_codename}_single": f"{cpc_all_name}: Single",
    f"{cpca_id_td_codename}_single": f"{cpc_all_name}+ID+TD: Single",
    f"{cpca_id_td_codename}_average": f"{cpc_all_name}+ID+TD: Average",
    f"{cpca_id_td_codename}_soft": f"{cpc_all_name}+ID+TD: Softmax",
    # f"{cpca_id_td_codename}_attn-e": f"{cpc_all_name}+ID+TD: Attn+E",
    f"{cpca_id_td_codename}_attn-2e": f"{cpc_all_name}+ID+TD: Attn+E",
    f"{cpca_id_td_codename}_attn": f"{cpc_all_name}+ID+TD: Attn",
    # "baseline_ddppo": "Baseline DDPPO",
    # f"{cpca_id_td_codename}_single_ddppo": f"{cpc_all_name}+ID+TD: Single DDPPO",
    # f"{cpca_id_td_codename}_attn-2e_ddppo": f"{cpc_all_name}+ID+TD: Attn+E DDPPO",
}

def get_run_logs(v):
    folder = os.path.join(run_root, v)
    run_folders = os.listdir(folder)
    run_folders.sort()
    event_paths = []
    for run_folder in run_folders:
        if 'run' in run_folder:
            full_path = os.path.join(folder, run_folder)
            event_paths.append(full_path)
    return event_paths


tf_size_guidance = {'scalars': 1000}
plot_key_folder_dict = {
    'eval_spl': 'eval_metrics_spl/',
    'eval_success': 'eval_metrics_success/'
}

#%%
run_root = "/nethome/jye72/projects/habitat-pointnav-aux/tb/r3/"
# run_root = "/nethome/jye72/projects/habitat-pointnav-aux/tb/mp3d_pn/"
run_count = 4
np.random.seed(0)
# nested by variant and then run i

# Set what to plot
variants_1 = ['baseline', f'{cpc_codename}16', 'id', 'td']
variants_2 = ['baseline', f'{cpc_codename}16', f"{cpc_codename}_single", f"{cpca_id_td_codename}_single"]
variants_3 = ['baseline', f"{cpc_codename}16", f"{cpca_id_td_codename}_soft", f"{cpca_id_td_codename}_attn-2e", f"{cpca_id_td_codename}_single"]
plotted_union = list(set(variants_1) | set(variants_2) | set(variants_3))
# plotted_union = [ "baseline", f"{cpca_id_td_codename}_attn-2e"]
# plotted_union = ["baseline", f"{cpca_id_td_codename}_single", f"{cpca_id_td_codename}_attn-2e"]
# plotted_union = [f"{cpca_id_td_codename}_attn-2e"]
# plotted_union = [f"{cpc_codename}_attn", f"{cpc_codename}_attn-e",f"{cpc_codename}_repeat",f"{cpc_codename}_fixed"]
# plotted_union = [f"{cpc_codename}_attn", f"{cpc_codename}_attn-e",f"{cpc_codename}_repeat",f"{cpc_codename}_fixed"]
# plotted_union = [f"{cpca_id_td_codename}_soft", f"{cpca_id_td_codename}_average", f"{cpc_codename}_single", f"{cpca_id_td_codename}_attn"]
# plotted_union = [f'{cpc_codename}1', f'{cpc_codename}2', f'{cpc_codename}4',]
# plotted_union = [f'{cpc_codename}16w', 'id', 'td']

palette = sns.color_palette(palette='muted', n_colors=len(plotted_union), desat=0.9)

variants = plotted_union
variants = variant_labels.keys()
variants = ['baseline', 'cpca-id-td_single', 'cpca-id-td_attn-2e']
variant_colors = {}
for i, v in enumerate(plotted_union):
    variant_colors[v] = palette[(i+3) % len(plotted_union)]

sns.palplot(palette)
variant_paths = {}
for variant in variants:
    variant_paths[variant] = get_run_logs(variant)

#%%
# * Key
# plot_key = 'success' # spl, success, eval_spl, eval_success
# plot_key = 'spl' # spl, success, eval_spl, eval_success
plot_key = 'eval_success' # spl, success, eval_spl, eval_success
plot_key = 'eval_spl' # spl, success, eval_spl, eval_success

plot_key_folder = plot_key_folder_dict.get(plot_key, "")

# Load
plot_values = {}
plot_steps = {}
for variant, variant_runs in variant_paths.items():
    plot_values[variant] = []
    plot_steps[variant] = []
    min_steps = 0
    for i, run in enumerate(variant_runs):
        if len(plot_steps[variant]) >= run_count:
            break
        accum_path = os.path.join(run, plot_key_folder)
        if not os.path.exists(accum_path):
            continue
        event_acc = EventAccumulator(accum_path, tf_size_guidance)
        event_acc.Reload()
        scalars = event_acc.Scalars('eval_metrics')
        steps_and_values = np.stack(
            [np.asarray([scalar.step, scalar.value])
            for scalar in scalars])
        steps = steps_and_values[:, 0]
        values = steps_and_values[:, 1]
        if len(steps) < 41: # We allow more in case we doubled something
            print(f"skipping {variant}, {i}")
            unique, indices = np.unique(steps, return_index=True)
            print(unique)
            print(values[-1])
            continue # Incomplete
        plot_steps[variant].append(steps)
        plot_values[variant].append(values)
    # print(variant)
    # for run in plot_values[variant]:
    #     print(len(run))
#%%
# * Cropping (and averaging) values of each checkpoint - for multi-eval
def get_cleaned_data(raw_steps, raw_values, average=1):
    clean_steps = {}
    clean_values = {}
    for variant in variants:
        clean_steps[variant] = []
        clean_values[variant] = []
        if variant in plot_steps:
            for i in range(len(plot_steps[variant])):
                steps = raw_steps[variant][i]
                vals = raw_values[variant][i]
                un, ind, inv = np.unique(steps, return_index=True, return_inverse=True)
                # all the places where there are 0s, is where the first unique is. Select them
                clean_steps[variant].append(steps[ind])
                avg_values = []
                for step in range(len(un)):
                    step_vals = vals[inv == step][:average]
                    # print(step, len(step_vals))
                    avg_step_val = np.mean(step_vals)
                    avg_values.append(avg_step_val)
                clean_values[variant].append(avg_values)
    return clean_steps, clean_values
clean_steps, clean_values = get_cleaned_data(plot_steps, plot_values, average=3)
#%%
best_ckpts = {}
for variant in clean_values:
    if len(clean_values[variant]) == 4 and len(clean_values[variant][3]) < 40:
        print(variant)
        var_data = np.array(clean_values[variant][:3])
    else:
        var_data = np.array(clean_values[variant])
        best_ckpt = 2 * (np.argmax(var_data, axis=1))
    best_ckpts[variant] = best_ckpt.tolist()
    print(f"{variant:20} {best_ckpts[variant]}")

import json
with open(f"{plot_key}_ckpts.csv", 'w') as f:
    json.dump(best_ckpts, f)

#%%
print(clean_values['baseline'][0][-1])
print(clean_values['cpca-id-td_single'][1][-3])
print(clean_values['cpca-id-td_attn-2e'][0][-3])
#%%
def get_means_and_ci(values, window_size=1, early_stop=True):
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
        # data = np.array(values[variant])
        min_overlap = min(len(trial) for trial in values[variant])
        data = np.array([trial[:min_overlap] for trial in values[variant]])
        # print(data.shape)
        # print(variant)
        values_smoothed = np.empty_like(data)
        if window_size > 1:
            for i in range(data.shape[1]):
                window_start = max(0, i - window_size)
                window = data[:, window_start:i + 1]
                values_smoothed[:, i] = window.mean(axis=1)
        else:
            values_smoothed = data

        if early_stop:
            best_until = np.copy(values_smoothed)
            for t in range(best_until.shape[1]):
                best_until[:,t] = np.max(best_until[:,:t+1], axis=1)
            values_smoothed = best_until
        means[variant] = np.mean(values_smoothed, axis=0)
        ci[variant] = 1.96 * np.std(values_smoothed, axis=0) \
            / math.sqrt(run_count) # 95%
    return means, ci

if 'eval' in plot_key:
    # data = plot_values
    data = clean_values
else:
    data = interpolated_values

plot_means, plot_ci = get_means_and_ci(data, window_size=1, early_stop=True)
true_means, true_ci = get_means_and_ci(data, window_size=1, early_stop=False) # For AUC calc


#%%
print(clean_values['cpca-id-td_attn-2e'][0][-1])
print(clean_values['cpca-id-td_attn-2e'][1][-1])
print(clean_values['cpca-id-td_attn-2e'][2][-1])
print(clean_values['cpca-id-td_attn-2e'][3][-1])
print(plot_means['cpca-id-td_attn-2e'][-1])
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
plt.xlim(0, 40)
plt.xticks(np.arange(0, 45, 5))
x_scale = 1e6

if 'eval' in plot_key:
    lower_lim = 0.0
    # upper_lim = 0.5 if 'success' in plot_key else .3
    upper_lim = 0.9 if 'success' in plot_key else .8

    plt.ylim(lower_lim, upper_lim)
    plt.yticks(np.arange(lower_lim, upper_lim + 0.01, 0.1))

# * Plot settings
set_num = 2
variant_lists = [variants_1, variants_2, variants_3]
plotted = variant_lists[set_num]
# plotted = ['baseline', 'cpca4', 'cpca-id-td_soft', 'cpca-id-td_single', 'cpca-id-td_attn', 'cpca-id-td_attn-e']

# Table 1
# plotted = ['baseline', 'cpca-id-td_attn-2e']

# plotted = ['baseline', 'cpca4', 'cpca-id-td_soft', 'cpca-id-td_attn-2e']

# plotted = ['baseline', 'cpca-id-td_soft', 'cpca-id-td_attn', 'cpca-id-td_attn-2e', 'cpca_single', 'cpca-id-td_single']
# plotted = variants
for variant in plotted:
    if 'eval' in plot_key:
        x = clean_steps[variant][0] / x_scale
    y = plot_means[variant]
    line, = plt.plot(x, y, label=variant_labels.get(variant, variant), c=variant_colors.get(variant))
    plt.fill_between(x, y - plot_ci[variant], y + plot_ci[variant], facecolor=line.get_color(), alpha=0.5)

def annotate(idx, from_var, to_var, hoffset=-6, voffset=0):
    lo = plot_means[from_var][idx]
    hi = plot_means[to_var][idx]
    if (hi - lo) > 0:
        sign = "+"

    else:
        sign = "-"
    plt.text(idx+hoffset, hi+voffset, f"{sign} {abs(hi - lo):.2f}", size=16)
    plt.annotate("", xy=(idx, lo), xycoords="data", xytext=(idx, hi), textcoords="data", arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0", linewidth="1.5"))

# Simple
if set_num == 0:
    annotate(40, "baseline", "cpca16", hoffset=-6.5, voffset=0.02)
    # annotate(2, "baseline", "cpca16", hoffset=1, voffset=0.02)
    leg_start = .71

# Homo
if set_num == 1:
    # annotate(40, "baseline", "cpca16", -6, -0.08)
    annotate(40, "baseline", "cpca-id-td_single", -6, 0.02)
    annotate(2, "baseline", "cpca-id-td_single", 2, 0.05)
    leg_start = .36
    # leg_start = .57

# Diverse
if set_num == 2:
    leg_start = .32
    annotate(2, "baseline", "cpca-id-td_attn-2e", 1.0, .01)

leg = plt.legend(loc=(leg_start, .01),
    markerfirst=False, ncol=1, frameon=False, labelspacing=0.4)
# leg = plt.legend(loc=(0.01, .7),
#     markerfirst=True, ncol=1, frameon=False, labelspacing=0.4)

for line in leg.get_lines():
    line.set_linewidth(2.0)

# plt.title("MP3D + Noisy Actuation + Sliding Off")
plt.savefig('test.pdf', dpi=150)

#%%
print(plot_means['baseline'][-1])
print(plot_means['cpca16'][19])
print(plot_means['cpca-id-td_single'][12])

#%%
# Teaser
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

plt.xlabel("Steps (Million)")
plt.ylabel("SPL (Higher is Better)")
spine_alpha = 0.3
plt.gca().spines['right'].set_alpha(0.0)
plt.gca().spines['bottom'].set_alpha(spine_alpha)
plt.gca().spines['left'].set_alpha(spine_alpha)
plt.gca().spines['top'].set_alpha(0.0)
plt.grid(alpha=0.25)
plt.tight_layout()

# Plot evals
# Axes
plt.xlim(0, 40)
plt.xticks(np.arange(0, 45, 5))
x_scale = 1e6

if 'eval' in plot_key:
    lower_lim = 0.0
    # upper_lim = 0.5 if 'success' in plot_key else .3
    upper_lim = 0.9 if 'success' in plot_key else .8

    plt.ylim(lower_lim, upper_lim)
    plt.yticks(np.arange(lower_lim, upper_lim + 0.01, 0.1))

# * Plot settings
variant_labels = {
    'baseline': "DD-PPO (Wijmans et al., 2020)",
    "cpca-id-td_attn-2e": "Ours"
}
plotted = ['cpca-id-td_attn-2e', 'baseline']
for variant in plotted:
    if 'eval' in plot_key:
        x = clean_steps[variant][0] / x_scale
    y = plot_means[variant]
    line, = plt.plot(x, y, label=variant_labels.get(variant, variant), c=variant_colors.get(variant))
    plt.fill_between(x, y - plot_ci[variant], y + plot_ci[variant], facecolor=line.get_color(), alpha=0.5)

idx = 40
hoffset = -10
voffset = -0.1
lo = plot_means['baseline'][idx]
hi = plot_means['cpca-id-td_attn-2e'][idx]

plt.annotate("", xy=(idx, lo), xycoords="data", xytext=(idx, hi + 0.01), textcoords="data", arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0", linewidth="1.5"))
plt.text(idx+hoffset, hi+voffset, f"+{(hi - lo):.2f} SPL", size=16)

plt.annotate("", xy=(40, lo), xycoords="data", xytext =(7, lo), textcoords="data", arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0", linewidth="1.5"))
plt.text(18, lo + 0.02, f"5.5x faster", size=16)

leg = plt.legend(loc=(0.32, .05), markerfirst=False, ncol=1, frameon=False, labelspacing=0.4)

for line in leg.get_lines():
    line.set_linewidth(2.0)

plt.title("Performance on PointGoal Navigation \n (with RGB + GPS + Compass sensors)")
plt.savefig('test.pdf', dpi=150)

#%%
# Hack around loading spl and success
#%%
# Prints values for tables
print(plot_key)
latex_label = {
    "baseline": "Baseline",
    "id": "ID",
    "td": "TD",
    "cpca1": "\cpcat$1$",
    "cpca2": "\cpcat$2$",
    "cpca4": "\cpcat$4$",
    "cpca8": "\cpcat$8$",
    "cpca16": "\cpcat$16$",
    "cpca16w": "Weighted \cpcat16",
    "cpca_single": "\\allcpc: Add",
    "cpca-id-td_single": "\\allcpc+ID+TD: Add",
    "cpca-id-td_attn-2e": "\\allcpc+ID+TD: Attn+E",
    "cpca-id-td_attn": "\\allcpc+ID+TD: Attn",
    "cpca-id-td_soft": "\\allcpc+ID+TD: Softmax",
    "cpca-id-td_average": "\\allcpc+ID+TD: Average",
    "cpca_attn": "\\allcpc: Attn",
    "cpca_attn-e": "\\allcpc: Attn+E",
    "cpca_fixed": "\\allcpc: Fixed Attn",
    "cpca_fixed": "\\allcpc: Fixed Attn",
    "cpca_repeat": "\cpcat16$\\times 5$: Attn",
}

basic_template = "\\rownumber {} & \n ${:.3f} $\scriptsize{{$\pm {:.3f}$}} & ${:.3f} $\scriptsize{{$\pm {:.3f}$}}"
# variant, auc, auc ci, best, best ci

for variant in variants:
    auc = metrics.auc(np.arange(0,1 + 1.0/40, 1.0/40), true_means[variant])
    auc_ci = metrics.auc(np.arange(0,1 + 1.0/40,1.0/40), true_ci[variant])
    print(basic_template.format(
        latex_label[variant],
        auc, auc_ci,
        plot_means[variant][-1], plot_ci[variant][-1]
    ))
    print("\\\\")
    # print(f"${auc:.3f} \pm {auc_ci:.3f}$")
    # print( # f"10M: ${plot_means[variant][ten_mil_key]:.3f} \pm {plot_ci[variant][ten_mil_key]:.3f}$ \n" +
    # f"${plot_means[variant][-1]:.3f} \pm {plot_ci[variant][-1]:.3f}$")

#%%
print(plot_means['cpca-id-td_attn-2e'])