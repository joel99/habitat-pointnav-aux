#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Detailed runner for collecting diagnostics for analysis. Additional tooling for map viz.
import os
import random

import numpy as np

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ppo.ppo_trainer import Diagnostics
from habitat_baselines.run import get_parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    run_exp(**vars(args))

# Just use diagnostics.basic for t-test, I believe

def run_exp(exp_config: str, run_type: str, ckpt_path="", run_id=None, run_suffix=None, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        ckpt_path: If evaluating, path to a checkpoint.
        run_id: If using slurm batch, run id to prefix.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    if run_type != "eval":
        print("Detailed runs only supported for evaluation")
        exit(1)

    config = get_config(exp_config, opts)
    variant_name = os.path.split(exp_config)[1].split('.')[0]
    config.defrost()
    if run_suffix != "" and run_suffix is not None:
        variant_name = f"{variant_name}-{run_suffix}"
    config.TENSORBOARD_DIR = os.path.join(config.TENSORBOARD_DIR, variant_name)
    config.CHECKPOINT_FOLDER = os.path.join(config.CHECKPOINT_FOLDER, variant_name)
    config.LOG_FILE = os.path.join(config.LOG_FILE, f"{variant_name}.log") # actually a logdir
    config.NUM_PROCESSES = 1

    run_prefix = 'run'
    if run_id is not None:
        config.TASK_CONFIG.SEED = run_id
        run_prefix = f'run_{run_id}'
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)

    # Sample input - /baseline/run_0.80.pth - need to add the extra folder
    ckpt_dir, ckpt_file = os.path.split(ckpt_path)
    variant = ckpt_dir.split('/')[-1]
    ckpt_index = ckpt_file.split('.')[1]
    ckpt_path = os.path.join(ckpt_dir, run_prefix, ckpt_file)

    # This config isn't used for detailed statistics, just put there somewhere where they won't overwrite

    # * Modify as desired
    detail_dir = os.path.join("/srv/share/jye72/r2_viz", "map_viz", f"{variant}_{run_prefix}_{ckpt_index}")
    config.VIDEO_DIR = os.path.join(detail_dir)

    # * Modify
    make_background = False
    map_name = "quantico"
    eval_stats_dir = os.path.join(f'/nethome/jye72/share/r3_detailed/') # /{map_name}')
    config.TASK_CONFIG.TASK_SENSORS = [
        'POINTGOAL_WITH_GPS_COMPASS_SENSOR',
        'GPS_SENSOR', 'HEADING_SENSOR'
    ]

    if make_background:
        config.TEST_EPISODE_COUNT = 1
        label = f"{variant}_{ckpt_index}_{run_prefix}-bg"
        log_diagnostics = [Diagnostics.basic, Diagnostics.top_down_map]
    else:
        config.VIDEO_OPTION = []
        label = f"{variant}_{ckpt_index}_{run_prefix}"
        # log_diagnostics = [Diagnostics.basic, Diagnostics.actions,
            # Diagnostics.weights, Diagnostics.gps, Diagnostics.heading]
        log_diagnostics = [Diagnostics.basic] # , Diagnostics.actions, Diagnostics.weights]
        log_diagnostics = [Diagnostics.basic, Diagnostics.actions, Diagnostics.weights]

    # * Modify as desired
    use_own_dataset = False

    if use_own_dataset:
        config.TASK_CONFIG.DATASET.DATA_PATH = \
            '/nethome/jye72/projects/data/datasets/pointnav/gibson/scene_viz/{split}/{split}.json.gz'
        config.TASK_CONFIG.DATASET.SPLIT = 'all'
        config.TASK_CONFIG.DATASET.SCENES_DIR = "data/scene_datasets/gibson"
        config.TASK_CONFIG.EVAL.SPLIT = 'all' # quirk in the code necessitates this

    # Make a top-down-map clean for visualization
    map_cfg = config.TASK_CONFIG.TASK.TOP_DOWN_MAP
    map_cfg.MAP_RESOLUTION = 4000
    map_cfg.DRAW_SOURCE = False
    map_cfg.DRAW_GOAL_POSITIONS = False
    map_cfg.DRAW_VIEW_POINTS = False
    map_cfg.DRAW_SHORTEST_PATH = False

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    trainer.eval(ckpt_path, log_diagnostics=log_diagnostics,
        output_dir=eval_stats_dir, label=label)

if __name__ == "__main__":
    main()
