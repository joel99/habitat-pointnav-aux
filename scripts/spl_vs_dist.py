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
eval_stat_root = '/nethome/jye72/share/r3_detailed/'

def get_data(variant, ckpt, run_id=0):
    eval_fn = f"{variant}_{ckpt}_run_{run_id}.json"
    log_path = osp.join(eval_stat_root, eval_fn)
    with open(log_path, 'r') as f:
        data = json.load(f)
    meta_df = []
    for i, ep in enumerate(data):
        episode_info = ep["episode_info"]
        info = ep['info']
        stats = ep['stats']
        basic, *_ = info.values()
        ep_id, scene_id, start_pos, start_rot, more_ep_info, _path_cache, goals, start_room, shortest_paths = episode_info.values()
        geo_dist = more_ep_info['geodesic_distance']
        reward, distance_to_goal, success, spl = stats.values()
        meta_df.append({
            "ep": ep_id,
            "scene": scene_id,
            "spl": spl,
            "success": success,
            "geo_dist": geo_dist,
        })

    meta_df = pd.DataFrame(meta_df)
    return meta_df

baseline_dfs = [
    get_data("baseline", 80, 0),
    get_data("baseline", 80, 1),
    get_data("baseline", 68, 2),
    get_data("baseline", 60, 3),
    # get_data("baseline", 76, 0),
]
baseline_df = pd.concat(baseline_dfs)
ours_dfs = [
    get_data("cpca-id-td_attn-2e", 76, 0),
    get_data("cpca-id-td_attn-2e", 66, 1),
    get_data("cpca-id-td_attn-2e", 68, 2),
    get_data("cpca-id-td_attn-2e", 72, 3),
    # get_data("cpca-id-td_attn-2e", 56, 1)
]
our_df = pd.concat(ours_dfs)

#%%
# print(our_df[our_df["geo_dist"] > 18])
print(len(baseline_df))
print(len(baseline_df[(baseline_df["geo_dist"] > 18) & (baseline_df["success"] == 1.0)]))
print(len(our_df[(our_df["geo_dist"] > 18) & (our_df["success"] == 1.0)]))

#%%
sns.set(style="whitegrid", font_scale=1.5)
data = dict(x=[], y=[], t=[])
names = dict(
    baseline="Baseline",
    ours="CPC|A{1-16}+ID+TD:Attn+E"
)
dfs = dict(
    baseline=baseline_df,
    ours=our_df
)
geo = baseline_df['geo_dist']

# bins = np.arange(geo.min(), geo.max() + 3, 3)
bins = np.array([0, 5, 10, 15, 25]) # np.arange(0, 26, 5)
print(bins)
for variant in dfs.keys():
    for i, ep in dfs[variant].iterrows():
        geo = ep["geo_dist"]
        idx = np.searchsorted(bins, geo, "right")
        geo = (bins[idx] + bins[idx - 1]) / 2.0
        data["x"].append(geo)
        data["y"].append(ep["spl"])
        data["t"].append(names[variant])
data = pd.DataFrame.from_dict(data)
hist_color = np.array([127, 140, 141, 127]) / 255
palette = [np.array([22, 160, 133]) / 255, np.array([230, 126, 34]) / 255]
g = (
    sns.FacetGrid(data, col="t", hue="t", height=6, palette=palette)
    .map(plt.hist, "x", density=True, color=tuple(hist_color.tolist()), bins=bins)
    .map_dataframe(sns.lineplot, "x", "y", markers=True, dashes=False)
    .despine()
    .set_titles("{col_name}")
    .set_axis_labels(
        "GeodesicDistance(Start, Goal)", "Performance (SPL; Higher is better)"
    )
)
plt.subplots_adjust(top=0.8)

g.savefig("spl-vs-geo.pdf", format="pdf", bbox_inches="tight")