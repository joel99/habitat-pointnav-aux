#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional
import random
import json
import attr

import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.auxiliary_tasks import get_aux_task_class
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
)
from habitat_baselines.rl.ppo import PPO, POLICY_CLASSES, MULTIPLE_BELIEF_CLASSES

# TODO support generic sensors
# TODO add compass
class Diagnostics:
    basic = "basic" # dummy to record episode stats (for t-test)
    actions = "actions"
    gps = "gps"
    heading = "heading"
    weights = "weights"
    top_down_map = "top_down_map"

@baseline_registry.register_trainer(name="ppo")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")
            self.checkpoint_prefix = config.TENSORBOARD_DIR.split('/')[-1]

        self._static_encoder = False
        self._encoder = None

    def get_ppo_class(self):
        return PPO

    def _setup_auxiliary_tasks(self, aux_cfg, ppo_cfg, task_cfg, observation_space=None, is_eval=False):
        aux_task_strings = [task.lower() for task in aux_cfg.tasks]
        # Differentiate instances of tasks by adding letters
        aux_counts = {}
        for i, x in enumerate(aux_task_strings):
            if x in aux_counts:
                aux_task_strings[i] = f"{aux_task_strings[i]}_{aux_counts[x]}"
                aux_counts[x] += 1
            else:
                aux_counts[x] = 1

        logger.info(f"Auxiliary tasks: {aux_task_strings}")

        num_recurrent_memories = 1
        if ppo_cfg.policy in MULTIPLE_BELIEF_CLASSES:
            num_recurrent_memories = len(aux_cfg.tasks)

        init_aux_tasks = []
        if not is_eval:
            for task in aux_cfg.tasks:
                task_class = get_aux_task_class(task)
                aux_module = task_class(ppo_cfg, aux_cfg[task], task_cfg, self.device, observation_space=observation_space).to(self.device)
                init_aux_tasks.append(aux_module)

        return init_aux_tasks, num_recurrent_memories, aux_task_strings

    def _setup_actor_critic_agent(self, ppo_cfg: Config, task_cfg: Config, aux_cfg: Config = None, aux_tasks=[]) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        if ppo_cfg.policy not in POLICY_CLASSES:
            raise Exception(f"Illegal policy {ppo_cfg.policy} provided. Valid policies are {POLICY_CLASSES.keys()}")
        if len(aux_tasks) != 0 and len(aux_tasks) != len(aux_cfg.tasks):
            raise Exception(f"Policy specifies {len(aux_cfg.tasks)} tasks but {len(aux_tasks)} were initialized.")
        policy_class = POLICY_CLASSES[ppo_cfg.policy]

        # Default policy settings for object nav
        is_objectnav = "ObjectNav" in task_cfg.TYPE
        additional_sensors = []
        embed_goal = False
        if is_objectnav:
            additional_sensors = ["gps", "compass"]
            embed_goal = True

        self.actor_critic = policy_class(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            aux_tasks=aux_tasks,
            goal_sensor_uuid=task_cfg.GOAL_SENSOR_UUID,
            num_tasks=len(aux_cfg.tasks), # we pass this is in to support eval, where no aux modules are made
            additional_sensors=additional_sensors,
            embed_goal=embed_goal,
            device=self.device,
            config=ppo_cfg.POLICY
        ).to(self.device)

        self.agent = self.get_ppo_class()(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            aux_loss_coef=ppo_cfg.aux_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            aux_tasks=aux_tasks,
            aux_cfg=aux_cfg,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        ).to(self.device)

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "optim_state": self.agent.optimizer.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.get_recurrent_states()[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        env_time += time.time() - t_step_env
        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)
        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.get_recurrent_states()[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy, aux_task_losses, aux_dist_entropy, aux_weights = self.agent.update(rollouts)

        rollouts.after_update()
        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
            aux_task_losses,
            aux_dist_entropy,
            aux_weights
        )

    def train(self, ckpt_path="", ckpt=-1, start_updates=0) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        ppo_cfg = self.config.RL.PPO
        task_cfg = self.config.TASK_CONFIG.TASK
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Initialize auxiliary tasks
        observation_space = self.envs.observation_spaces[0]
        aux_cfg = self.config.RL.AUX_TASKS
        init_aux_tasks, num_recurrent_memories, aux_task_strings = \
            self._setup_auxiliary_tasks(aux_cfg, ppo_cfg, task_cfg, observation_space)

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            observation_space,
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            num_recurrent_memories=num_recurrent_memories
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        self._setup_actor_critic_agent(ppo_cfg, task_cfg, aux_cfg, init_aux_tasks)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        if ckpt != -1:
            logger.info(
                f"Resuming runs at checkpoint {ckpt}. Timing statistics are not tracked properly."
            )
            assert ppo_cfg.use_linear_lr_decay is False and ppo_cfg.use_linear_clip_decay is False, "Resuming with decay not supported"
            # This is the checkpoint we start saving at
            count_checkpoints = ckpt + 1
            count_steps = start_updates * ppo_cfg.num_steps * self.config.NUM_PROCESSES
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            self.agent.load_state_dict(ckpt_dict["state_dict"])
            if "optim_state" in ckpt_dict:
                self.agent.optimizer.load_state_dict(ckpt_dict["optim_state"])
            else:
                logger.warn("No optimizer state loaded, results may be funky")
            if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
                count_steps = ckpt_dict["extra_state"]["step"]


        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:

            for update in range(start_updates, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                for step in range(ppo_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                delta_pth_time, value_loss, action_loss, dist_entropy, aux_task_losses, aux_dist_entropy, aux_weights = self._update_agent(
                    ppo_cfg, rollouts
                )
                pth_time += delta_pth_time

                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "entropy",
                    dist_entropy,
                    count_steps,
                )

                writer.add_scalar(
                    "aux_entropy",
                    aux_dist_entropy,
                    count_steps
                )

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], count_steps
                )

                # Check to see if there are any metrics
                # that haven't been logged yet
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(metrics) > 0:
                    writer.add_scalars("metrics", metrics, count_steps)

                losses = [value_loss, action_loss] + aux_task_losses
                writer.add_scalars(
                    "losses",
                    {k: l for l, k in zip(losses, ["value", "policy"] + aux_task_strings)},
                    count_steps,
                )

                writer.add_scalars(
                    "aux_weights",
                    {k: l for l, k in zip(aux_weights, aux_task_strings)},
                    count_steps,
                )

                writer.add_scalar(
                    "success",
                    deltas["success"] / deltas["count"],
                    count_steps,
                )

                # Log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tvalue_loss: {}\t action_loss: {}\taux_task_loss: {} \t aux_entropy {}".format(
                            update, value_loss, action_loss, aux_task_losses, aux_dist_entropy
                        )
                    )
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"{self.checkpoint_prefix}.{count_checkpoints}.pth", dict(step=count_steps)
                    )
                    count_checkpoints += 1

        self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
        log_diagnostics=[],
        output_dir='.',
        label='.',
        num_eval_runs=1
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if checkpoint_index == -1:
            ckpt_file = checkpoint_path.split('/')[-1]
            split_info = ckpt_file.split('.')
            checkpoint_index = split_info[1]
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO
        task_cfg = config.TASK_CONFIG.TASK

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        # pass in aux config if we're doing attention
        aux_cfg = self.config.RL.AUX_TASKS
        self._setup_actor_critic_agent(ppo_cfg, task_cfg, aux_cfg)

        # Check if we accidentally recorded `visual_resnet` in our checkpoint and drop it (it's redundant with `visual_encoder`)
        ckpt_dict['state_dict'] = {
            k:v for k, v in ckpt_dict['state_dict'].items() if 'visual_resnet' not in k
        }
        self.agent.load_state_dict(ckpt_dict["state_dict"])

        logger.info(
            "agent number of trainable parameters: {}".format(
                sum(
                    param.numel()
                    for param in self.agent.parameters()
                    if param.requires_grad
                )
            )
        )

        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        _, num_recurrent_memories, _ = self._setup_auxiliary_tasks(aux_cfg, ppo_cfg, task_cfg, is_eval=True)
        if self.config.RL.PPO.policy in MULTIPLE_BELIEF_CLASSES:
            aux_tasks = self.config.RL.AUX_TASKS.tasks
            num_recurrent_memories = len(self.config.RL.AUX_TASKS.tasks)
            test_recurrent_hidden_states = test_recurrent_hidden_states.unsqueeze(2).repeat(1, 1, num_recurrent_memories, 1)


        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )

        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]

        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        videos_cap = 2 # number of videos to generate per checkpoint
        if len(log_diagnostics) > 0:
            videos_cap = 10
        # video_indices = random.sample(range(self.config.TEST_EPISODE_COUNT),
            # min(videos_cap, self.config.TEST_EPISODE_COUNT))
        video_indices = range(10)
        print(f"Videos: {video_indices}")

        total_stats = []
        dones_per_ep = dict()

        # Logging more extensive evaluation stats for analysis
        if len(log_diagnostics) > 0:
            d_stats = {}
            for d in log_diagnostics:
                d_stats[d] = [
                    [] for _ in range(self.config.NUM_PROCESSES)
                ] # stored as nested list envs x timesteps x k (# tasks)

        pbar = tqdm.tqdm(total=number_of_eval_episodes * num_eval_runs)
        self.agent.eval()
        while (
            len(stats_episodes) < number_of_eval_episodes * num_eval_runs
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()
            with torch.no_grad():
                weights_output = None
                if self.config.RL.PPO.policy in MULTIPLE_BELIEF_CLASSES:
                    weights_output = torch.empty(self.envs.num_envs, len(aux_tasks))
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                    weights_output=weights_output
                )
                prev_actions.copy_(actions)

                for i in range(self.envs.num_envs):
                    if Diagnostics.actions in log_diagnostics:
                        d_stats[Diagnostics.actions][i].append(prev_actions[i].item())
                    if Diagnostics.weights in log_diagnostics:
                        aux_weights = None if weights_output is None else weights_output[i]
                        if aux_weights is not None:
                            d_stats[Diagnostics.weights][i].append(aux_weights.half().tolist())

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                next_k = (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                )
                if dones_per_ep.get(next_k, 0) == num_eval_runs:
                    envs_to_pause.append(i) # wait for the rest

                if not_done_masks[i].item() == 0:
                    episode_stats = dict()

                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )

                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats

                    k = (
                        current_episodes[i].scene_id,
                        current_episodes[i].episode_id,
                    )
                    dones_per_ep[k] = dones_per_ep.get(k, 0) + 1

                    if dones_per_ep.get(k, 0) == 1 and len(self.config.VIDEO_OPTION) > 0 and len(stats_episodes) in video_indices:
                        logger.info(f"Generating video {len(stats_episodes)}")
                        category = getattr(current_episodes[i], "object_category", "")
                        if category != "":
                            category += "_"
                        try:
                            generate_video(
                                video_option=self.config.VIDEO_OPTION,
                                video_dir=self.config.VIDEO_DIR,
                                images=rgb_frames[i],
                                episode_id=current_episodes[i].episode_id,
                                checkpoint_idx=checkpoint_index,
                                metrics=self._extract_scalars_from_info(infos[i]),
                                tag=f"{category}{label}",
                                tb_writer=writer,
                            )
                        except Exception as e:
                            logger.warning(str(e))
                    rgb_frames[i] = []

                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                            dones_per_ep[k],
                        )
                    ] = episode_stats

                    if len(log_diagnostics) > 0:
                        diagnostic_info = dict()
                        for metric in log_diagnostics:
                            diagnostic_info[metric] = d_stats[metric][i]
                            d_stats[metric][i] = []
                        if Diagnostics.top_down_map in log_diagnostics:
                            top_down_map = torch.tensor([])
                            if len(self.config.VIDEO_OPTION) > 0:
                                top_down_map = infos[i]["top_down_map"]["map"]
                                top_down_map = maps.colorize_topdown_map(
                                    top_down_map, fog_of_war_mask=None
                                )
                            diagnostic_info.update(dict(top_down_map=top_down_map))
                        total_stats.append(
                            dict(
                                stats=episode_stats,
                                did_stop=bool(prev_actions[i] == 0),
                                episode_info=attr.asdict(current_episodes[i]),
                                info=diagnostic_info,
                            )
                        )
                    pbar.update()

                # episode continues
                else:
                    if len(self.config.VIDEO_OPTION) > 0:
                        aux_weights = None if weights_output is None else weights_output[i]
                        frame = observations_to_image(observations[i], infos[i], current_episode_reward[i].item(), aux_weights, aux_tasks)
                        rgb_frames[i].append(frame)
                    if Diagnostics.gps in log_diagnostics:
                        d_stats[Diagnostics.gps][i].append(observations[i]["gps"].tolist())
                    if Diagnostics.heading in log_diagnostics:
                        d_stats[Diagnostics.heading][i].append(observations[i]["heading"].tolist())

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)
            logger.info("eval_metrics")
            logger.info(metrics)
        if len(log_diagnostics) > 0:
            os.makedirs(output_dir, exist_ok=True)
            eval_fn = f"{label}.json"
            with open(os.path.join(output_dir, eval_fn), 'w', encoding='utf-8') as f:
                json.dump(total_stats, f, ensure_ascii=False, indent=4)
        self.envs.close()
