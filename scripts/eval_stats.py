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
eval_stat_root = '/nethome/jye72/share/eval_stats/'

#%%
variant = 'fpcit_attn-e'
aux_tasks = ['cpc|a-1', 'cpc|a-2', 'cpc|a-4', 'cpc|a-8', 'cpc|a-16', 'ID', 'TD']
aux_tasks = [s.upper() for s in aux_tasks]
ckpt = 80
run_id = 'run_0' # 'pn_gc_ckpts' # lol, that's a bug

# eval_fn = f"{variant}_{ckpt}_run-{run_id}.json"
eval_fn = f"{run_id}_{ckpt}_run-{variant}.json"
log_path = osp.join(eval_stat_root, run_id, eval_fn)
with open(log_path, 'r') as f:
    data = json.load(f)

#%%
# Cross ep analysis
num_eps = len(data)

wiggle_count = 0

meta_df = pd.DataFrame(index=['ep', 'scene'], columns=['did_stop', 'steps', 'geo_dist', 'spl', 'success', 'rotate_streak', 'left', 'right', 'forward'])
for i, ep in enumerate(data):
    map_arr = np.asarray(ep['top_down_map'], dtype=np.uint8)
    info = ep['info']
    stats = ep['stats']
    episode_info, did_stop, actions, weights, gps = info.values()
    ep_id, scene_id, start_pos, start_rot, more_ep_info, goals, start_room, shortest_apth = episode_info.values()
    geo_dist = more_ep_info['geodesic_distance']
    spl, success, reward = stats.values()
    actions = np.array(actions)
    weights = np.array(weights)

    last_rot = -1
    streak = 0
    wiggle_streak = 0
    longest_streak = 0
    for act in actions:
        if act == LEFT_VAL and last_rot == LEFT_VAL or act == RIGHT_VAL and last_rot == RIGHT_VAL:
            streak += 1
            wiggle_streak += 1
        if act == LEFT_VAL and last_rot == RIGHT_VAL or act == RIGHT_VAL and last_rot == LEFT_VAL:
            last_rot = act
            wiggle_streak += 1
            if streak > longest_streak:
                longest_streak = streak
            streak = 1
        elif act == LEFT_VAL or act == RIGHT_VAL: # starting a streak
            last_rot = act
            streak = 1
            wiggle_streak += 1
            if streak > longest_streak:
                longest_streak = streak
        else: # non-turn
            last_rot = -1
            if streak > longest_streak:
                longest_streak = streak
            streak = 0
            if wiggle_streak > 10:
                wiggle_count += 1
            wiggle_streak = 0

    meta_df = meta_df.append({
        "ep": ep_id,
        "scene": scene_id,
        "spl": spl,
        "success": success,
        "did_stop": did_stop,
        "steps": len(actions),
        "geo_dist": geo_dist,
        "forward": np.count_nonzero(actions == FORWARD_VAL),
        "left": np.count_nonzero(actions == LEFT_VAL),
        "right": np.count_nonzero(actions == RIGHT_VAL),
        "rotate_streak": longest_streak,
    }, ignore_index=True)
    # did_rotate as the accumulation of multiple rotations in the same direction
    # weights = np.array(weights)
    # print(map_arr.shape)
    # plt.imshow(map_arr)

print(wiggle_count)
#%%
# Query - what percentage of episodes started with a lot of rotations?

# print(meta_df.describe(include='all'))
# print(len(meta_df))
print(meta_df.geo_dist)

print(len(meta_df[meta_df['rotate_streak'] > 22]))
# only 1 baseline run has this many rotations (failed)
# 18 fpcit_attn-e have this (w/ about 60% success)
meta_df[meta_df['rotate_streak'] > 22][['rotate_streak', 'ep', 'success', 'spl']]
# =====================================
# 452, 818,  781

#%%
# action - weights dist
steps = []
cols = ['action']
cols.extend(aux_tasks)
for ep in data:
    weights_total = np.zeros(7)
    map_arr = np.asarray(ep['top_down_map'], dtype=np.uint8)
    info = ep['info']
    stats = ep['stats']
    episode_info, did_stop, actions, weights, gps = info.values()
    ep_id, scene_id, start_pos, start_rot, more_ep_info, goals, start_room, shortest_apth = episode_info.values()
    episode_info, did_stop, actions, weights, gps = info.values()
    spl, success, reward = stats.values()
    for i, action in enumerate(actions):
        step = [action]
        step.extend(weights[i])
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