#%%
import numpy as np
import pandas as pd
import os
import os.path as osp
import json
import matplotlib.pyplot as plt
import seaborn as sns
import PIL.Image
from habitat.utils.visualizations import maps, utils
from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_from_coeff,
    quaternion_rotate_vector,
)

#%%
map_name = "quantico"
action_names = [
    "STOP",
    "FWD",
    "LEFT",
    "RIGHT"
]

STOP_VAL = 0
FORWARD_VAL = 1
LEFT_VAL = 2
RIGHT_VAL = 3
eval_stat_root = f'/nethome/jye72/share/map_viz/{map_name}'

# cantwell_x_bounds = (2329 - 3, 2722 + 3) # 3 = map padding
# cantwell_y_bounds = (1488 - 3, 1666 + 3) # 3 = map padding
padding = 3
map_x_bounds = (2249 - padding, 2526 + padding)
map_y_bounds = (1542 - padding, 1739 + padding)

CLIPPED_X, _ = map_x_bounds
CLIPPED_Y, _ = map_y_bounds

map_resolution = (4000, 4000)
COORDINATE_EPSILON = 1e-6
COORDINATE_MIN = -62.3241 - COORDINATE_EPSILON
COORDINATE_MAX = 90.0399 + COORDINATE_EPSILON


#%%
variant = 'fpcit_attn-e' # 'e-fpcit'
aux_tasks = ['cpc1', 'cpc2', 'cpc4', 'cpc8', 'cpc16', 'ID', 'TD']
colors = sns.color_palette("hls", len(aux_tasks))
sns.set()

ckpt = 80
run_id = 'run_0' # 'pn_gc_ckpts' # lol, that's a bug
cols = ['pos_x', 'pos_y', 'heading', 'action'] # action taken at time t, not to get to time t
cols.extend(aux_tasks)

eval_fn = f"{variant}_{ckpt}_run-{run_id}.json"
log_path = osp.join(eval_stat_root, variant, eval_fn)
with open(log_path, 'r') as f:
    data = json.load(f)

locs = []
for ep_num, ep in enumerate(data):
    info = ep['info']
    episode_info, start_rot, actions, weights, gps, heading = info.values()
    print(episode_info.keys())
    _, _, start_pos, start_rot, more_ep_info, goals, start_room, shortest_path = episode_info.values()
    # if not in_goal_box(goals[0]["position"]):
    #     continue
    for i, reading in enumerate(gps):
        reading_pos = [reading[1], start_pos[1], -reading[0]]
        # reading_pos = [reading[0], start_pos[1], reading[1]]
        head_phi = heading[i] # this is already global
        rotation_world_start = quaternion_from_coeff(start_rot)
        global_offset = quaternion_rotate_vector(
            rotation_world_start, reading_pos
        )
        global_pos = global_offset + start_pos
        cur_x, _, cur_y = global_pos
        loc_info = [cur_x, cur_y, head_phi, actions[i]]
        loc_info.extend(weights[i])
        locs.append(loc_info)

df = pd.DataFrame(locs, columns=cols, index=[i for i in range(len(locs))])
for task in aux_tasks:
    df[task] = pd.to_numeric(df[task])
df['head_cat'] = pd.cut(df['heading'], [-1 * np.pi, -1 * np.pi / 2, 0, np.pi/2, np.pi],# bins=dist_bins,
    labels=['N', 'W', 'S', 'E'])
df['head_cat'].describe()

#%%
# Load Map
eval_fn = f"{variant}_{ckpt}_run-{run_id}-bg.json"
log_path = osp.join(eval_stat_root, variant, eval_fn)
with open(log_path, 'r') as f:
    data_quick = json.load(f)
ep = data_quick[0]
info = ep['info']
stats = ep['stats']
episode_info, did_stop, actions, weights, gps, heading = info.values()
ep_id, scene_id, start_pos, start_rot, more_ep_info, goals, start_room, shortest_path = episode_info.values()
map_arr = np.asarray(ep['top_down_map'], dtype=np.uint8)
plt.imshow(map_arr)

goal = goals[0]["position"]
goal_radius = 1.0
def in_goal_box(coords, radius=goal_radius):
    return goal[0] - radius <= coords[0] <= goal[0] + radius and \
        goal[2] - radius <= coords[2] <= goal[2] + radius

#%%

box_size = (3, 3)
top_down_map = map_arr

# head_cat = 'S'
# for i, loc in df[df['head_cat'] == head_cat].iterrows():
ctr = 0
for i, loc in df.iterrows():
    # if ctr > 50000:
    #     break
    # else:
    #     ctr += 1

    max_weight_i = np.argmax(loc[aux_tasks])
    color = colors[max_weight_i]
    p_x, p_y = maps.to_grid(
        loc['pos_x'],
        loc['pos_y'],
        COORDINATE_MIN,
        COORDINATE_MAX,
        map_resolution,
    )
    a_x = p_x - CLIPPED_X
    a_y = p_y - CLIPPED_Y
    box = (np.ones((*box_size, 3)) * 255 * color).astype(np.uint8)
    utils.paste_overlapping_image(
        background=top_down_map,
        foreground=box,
        location=(a_x, a_y)
    )

a, b = 4, 4
n = 9
r = 3

y,x = np.ogrid[-a:n-a, -b:n-b]
mask = x*x + y*y <= r*r
goal_circle = np.zeros((n, n, 3))
goal_circle[mask] = 255
goal_center = maps.to_grid(
    goal[0], goal[2], COORDINATE_MIN, COORDINATE_MAX, map_resolution
)
goal_a_x, goal_a_y = goal_center[0] - CLIPPED_X, goal_center[1] - CLIPPED_Y
utils.paste_overlapping_image(
    background=top_down_map,
    foreground=goal_circle,
    location=(goal_a_x, goal_a_y))
dir_map = {'N': 'left', 'W': 'down', 'S': 'right', 'E': 'up'}
sns.set_palette(colors)
fig, axes = plt.subplots(2,gridspec_kw={'height_ratios': [4, 1]}, figsize=(7, 6))
if top_down_map.shape[0] > top_down_map.shape[1]:
    top_down_map = np.rot90(top_down_map, 1)
# fig.suptitle(f"Direction: {dir_map[head_cat]}")
axes[0].imshow(top_down_map)
axes[0].axis('off')
axes[1].yaxis.set_visible(False)
task_df = pd.DataFrame({"height": 1, "task": aux_tasks})
sns.barplot(x="task", y="height", data=task_df, ax=axes[1])
plt.savefig('map_viz.png', dpi=300)