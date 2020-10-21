#%%
import numpy as np
import pandas as pd
import os
import os.path as osp
import json
import matplotlib.pyplot as plt
import seaborn as sns
import PIL.Image
#%%
dash_styles = ["",
               (4, 1.5),
               (1, 1),
               (3, 1, 1.5, 1),
               (5, 1, 1, 1),
               (5, 1, 2, 1, 2, 1),
               (2, 2, 3, 1.5),
               (1, 2.5, 3, 1.2)]
action_names = [
    "STOP",
    "FWD",
    "LEFT",
    "RIGHT"
]

ROTATION_THRESH = 18
FORWARD_VAL = 1
LEFT_VAL = 2
RIGHT_VAL = 3
eval_stat_root = '/nethome/jye72/share/r3_detailed/'

#%%
variant = 'cpca-id-td_attn-2e'
# aux_tasks = ['cpc|a-1', 'cpc|a-2', 'cpc|a-4', 'cpc|a-8', 'cpc|a-16', 'ID', 'TD']
aux_tasks = ['cpc|a-1', 'cpc|a-2', 'cpc|a-4', 'cpc|a-8', 'cpc|a-16', 'id', 'td']
aux_tasks = [s.upper() for s in aux_tasks]
ckpt = 66
ckpt = 76
run_id = 'run_0' # 'pn_gc_ckpts' # lol, that's a bug

# eval_fn = f"{variant}_{ckpt}_run-{run_id}.json"
eval_fn = f"{variant}_{ckpt}_{run_id}.json"
log_path = osp.join(eval_stat_root, eval_fn)
with open(log_path, 'r') as f:
    data = json.load(f)

#%%
# Cross ep analysis
num_eps = len(data)

meta_df = pd.DataFrame(index=['ep', 'scene'], columns=['steps', 'geo_dist', 'spl', 'success', 'actions', 'weights'])
for i, ep in enumerate(data):
    info = ep['episode_info']
    stats = ep['stats']
    geodesic = info['info']['geodesic_distance']
    episode_id = info['episode_id']
    scene_id = info['scene_id']
    spl = stats['spl']
    success = stats['success']

    diag_info = ep['info']
    actions = diag_info['actions']
    weights = diag_info['weights']
    actions = np.array(actions)
    weights = np.array(weights)

    meta_df = meta_df.append({
        "ep": episode_id,
        "scene": scene_id,
        "spl": spl,
        "success": success,
        "geo_dist": geodesic,
        "actions": actions,
        "weights": weights,
    }, ignore_index=True)

print(meta_df['spl'].mean())
#%%
# action - weights dist
steps = []
cols = ['action']
cols.extend(aux_tasks)
for ep in data:
    weights_total = np.zeros(5)
    # map_arr = np.asarray(ep['top_down_map'], dtype=np.uint8)
    info = ep['info']
    stats = ep['stats']
    actions = info['actions']
    weights = info['weights']
    for i, action in enumerate(actions):
        step = [action]
        step.extend(weights[i])
        # print(weights)
        steps.append(step)

df = pd.DataFrame(steps, columns=cols)

#%%
# Aux task distribution for a given action
max_task = []
for i, step in df.iterrows():
    max_task.append(np.argmax(step[aux_tasks]))
df['max_task'] = max_task
#%%
# Unconditioned

plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
task_df = df
ax = sns.catplot(x="max_task", kind="count", data=task_df, ax=ax).axes.flatten()[0]
ax.set_xticklabels(aux_tasks)
ax.text(.5,.9, "ALL", fontsize=20,
    horizontalalignment='center',
    transform=ax.transAxes)
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
ax.set_ylabel("Steps")
ax.get_yaxis().set_visible(False)
ax.get_xaxis().get_label().set_visible(False)
ax.spines['left'].set_visible(False)
plt.savefig('task_dist.pdf',bbox_inches="tight")


#%%
fig, axes = plt.subplots(2, 2)
axes = axes.reshape(4)
for i, ax in enumerate(axes):
    plot = sns.catplot(x="max_task", kind="count", data=df[df['action'] == i], ax=ax)
    ax = plot.axes.flatten()[0]
    # ax.set_title(action_names[i], fontsize=20)
    ax.text(.5,.9, action_names[i], fontsize=20,
        horizontalalignment='center',
        transform=ax.transAxes)
    ax.set_xticklabels(aux_tasks)
    # ax.set_title(action_names[i])
    ax.set_ylabel("Steps")
    ax.set_xlabel("Task")
    ax.set_xticklabels(aux_tasks)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.set_ylabel("Steps")
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().get_label().set_visible(False)
    ax.spines['left'].set_visible(False)

    plot.savefig(f'aux_dist_{i}.pdf')


#%%
# Unconditioned
task_df = df
ax = sns.catplot(x="action", kind="count", data=task_df, ax=ax).axes.flatten()[0]
ax.set_xticklabels(action_names)
ax.set_ylabel("Steps")
ax.set_xlabel("Action")
ax.text(.7,.8, "ALL", fontsize=24,
        horizontalalignment='center',
        transform=ax.transAxes)
ax.set_ylabel("Steps")
ax.set_xlabel("Action")
ax.set_xticklabels(action_names)
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().get_label().set_visible(False)
ax.spines['left'].set_visible(False)
plt.savefig('action_dist.pdf',bbox_inches="tight")

#%%
# Action for a given aux task
plt.tight_layout()
new_columns = ['action']
new_columns.extend(aux_tasks)
new_columns.append('max_task')
df.columns = new_columns
fig, axes = plt.subplots(len(aux_tasks), figsize=(1,1))
for i, ax in enumerate(axes):
    task_df = df[df[aux_tasks[i]] > 0.1] # we're using it somewhat
    plot = sns.catplot(x="action", kind="count", data=task_df, ax=ax)
    ax = plot.axes.flatten()[0]
    ax.text(.7,.8, aux_tasks[i], fontsize=24,
        horizontalalignment='center',
        transform=ax.transAxes)
    ax.set_ylabel("Steps")
    ax.set_xlabel("Action")
    ax.set_xticklabels(action_names)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().get_label().set_visible(False)
    ax.spines['left'].set_visible(False)
    plot.savefig(f'action_dist_{i}.pdf')
#%%
# SPL binned by geodesic distance
# dist_bins = pd.IntervalIndex.from_tuples([(-0.1, 4.0), (4.0, 8.0), (8.0, 15.0), (15.0, 25.0)])
# print(meta_df['geo_dist'].max())
meta_df['path_class'] = pd.cut(meta_df['geo_dist'], [0, 4.0, 8.0, 15.0, 25.0],# bins=dist_bins,
    labels=['short', 'medium', 'long', 'extralong'])

# print(meta_df['path_class'])
sns.catplot(x='path_class', y='geo_dist', data=meta_df)

g = sns.FacetGrid(meta_df, col="path_class", col_wrap=2, margin_titles=True)
g.map(sns.distplot, "spl", color="steelblue", bins=5, rug="true")

# sns.distplot(meta_df[meta_df['geo_dist'] < 1], bins=5)

#%%
ep_index = 452
ep = None
for i in data:
    if i['info']['episode_info']['episode_id'] == ep_index:
        ep = i
ep = data[ep_index] # single data case study
map_arr = np.asarray(ep['top_down_map'], dtype=np.uint8)
info = ep['info']
stats = ep['stats']

episode_info, did_stop, actions, weights, gps = info.values()
ep_id, scene_id, start_pos, start_rot, more_ep_info, goals, start_room, shortest_apth = episode_info.values()
episode_info, did_stop, actions, weights = info.values()
spl, success, reward = stats.values()
# print(actions)
actions = np.array(actions)
weights = np.array(weights)
plt.axis('off')
plt.imshow(map_arr)

# weights = np.array(weights)
# print(map_arr.shape)

#%%

#%%
# one_hot = np.eye(len(action_names), dtype=np.uint8)[actions]
timestep = np.arange(len(actions)) # actions is the value
actions_cat = pd.Categorical(actions, categories=[0, 1, 2, 3])

df = pd.DataFrame(data={"timestep": timestep, "action": actions_cat})
df['action'] = df['action'].cat.rename_categories(action_names)

# want to plot timestep as y
sns.catplot(x="timestep", y="action", data=df, jitter=False)\
    .set(title=f'Ep: {ep_id} Scene: {scene_id} -- {variant}.{ckpt}')
# %%
for i, task in enumerate(aux_tasks):
    df[task] = weights[:,i]
task_info = pd.melt(df, id_vars=["timestep", "action"],
                    value_vars=aux_tasks, var_name="Task",value_name="Weight")
sns.relplot(x='timestep', y='Weight', data=task_info, hue="Task", style="action")