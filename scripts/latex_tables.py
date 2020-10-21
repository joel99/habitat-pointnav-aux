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
variants = ['cpca-id-td_attn']
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
# plot_key = 'eval_spl' # spl, success, eval_spl, eval_success

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
# Hack around loading spl and success
spl_means, spl_ci = plot_means, plot_ci
spl_true_means, spl_true_ci = true_means, true_ci

#%%
success_means, success_ci = plot_means, plot_ci
success_true_means, success_true_ci = true_means, true_ci

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
    "depth": "Depth (Fine-tuned)",
    "depth_nt": "Depth (Frozen)"
}

auc_template = "\\rownumber {} & \n ${:.3f} $\scriptsize{{$\pm {:.3f}$}} & ${:.3f} $\scriptsize{{$\pm {:.3f}$}}"
# variant, auc, auc ci, best, best ci
best_template = "& &\n ${:.3f} $\scriptsize{{$\pm {:.3f}$}} & ${:.3f} $\scriptsize{{$\pm {:.3f}$}}"
for variant in variants:
# for variant in ['cpca_single', 'cpca-id-td_single']:
# for variant in ['cpca-id-td_average', 'cpca-id-td_soft', 'cpca-id-td_soft', 'cpca-id-td_attn-2e']:
# for variant in ['baseline', 'cpca16', 'cpca16w',  'cpca_single', 'cpca-id-td_attn-2e', 'cpca_attn', 'cpca_attn-e', 'cpca_fixed', 'cpca_repeat']:
    spl_auc = metrics.auc(np.arange(0,1 + 1.0/40, 1.0/40), spl_true_means[variant])
    success_auc = metrics.auc(np.arange(0,1 + 1.0/40, 1.0/40), success_true_means[variant])
    spl_auc_ci = metrics.auc(np.arange(0,1 + 1.0/40,1.0/40), spl_true_ci[variant])
    success_auc_ci = metrics.auc(np.arange(0,1 + 1.0/40,1.0/40), success_true_ci[variant])
    print(auc_template.format(
        latex_label[variant],
        success_auc, success_auc_ci,
        spl_auc, spl_auc_ci,
    ))
    print(best_template.format(
        success_means[variant][-1], success_ci[variant][-1],
        spl_means[variant][-1], spl_ci[variant][-1],
    ))
    print("\\\\")
    # print(f"${auc:.3f} \pm {auc_ci:.3f}$")
    # print( # f"10M: ${plot_means[variant][ten_mil_key]:.3f} \pm {plot_ci[variant][ten_mil_key]:.3f}$ \n" +
    # f"${plot_means[variant][-1]:.3f} \pm {plot_ci[variant][-1]:.3f}$")

#%%
print(plot_means['cpca-id-td_attn-2e'])