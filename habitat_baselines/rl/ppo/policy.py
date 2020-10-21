#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import numpy as np
import torch
import torch.nn as nn

from habitat_baselines.common.utils import CategoricalNet, Flatten
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
import habitat_baselines.rl.models.resnet as resnet
from habitat_baselines.rl.models.resnet import ResNetEncoder

GOAL_EMBEDDING_SIZE = 32

class Policy(nn.Module):
    def __init__(self, net, dim_actions, **kwargs):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        **kwargs
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, features, None, None, None


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class BaselinePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid=None,
        hidden_size=512,
        **kwargs,
    ):
        super().__init__(
            BaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
            ),
            action_space.n,
        )

class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass

class BaselineNet(Net):
    r"""Network which passes the input image through CNN and passes through RNN.
    """

    def __init__(
        self,
        observation_space,
        hidden_size,
        goal_sensor_uuid=None,
        additional_sensors=[] # low dim sensors corresponding to registered name
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self.additional_sensors = additional_sensors
        self._n_input_goal = 0
        self._n_input_goal = 0
        if goal_sensor_uuid is not None and goal_sensor_uuid != "no_sensor":
            self.goal_sensor_uuid = goal_sensor_uuid
            self._initialize_goal_encoder(observation_space)
        self._hidden_size = hidden_size

        resnet_baseplanes = 32
        backbone="resnet18"
        visual_resnet = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=False,
        )
        self.visual_encoder = nn.Sequential(
            visual_resnet,
            Flatten(),
            nn.Linear(
                np.prod(visual_resnet.output_shape), hidden_size
            ),
            nn.ReLU(True),
        )

        final_embedding_size = (0 if self.is_blind else self._hidden_size) + self._n_input_goal
        for sensor in additional_sensors:
            final_embedding_size += observation_space.spaces[sensor].shape[0]

        self.state_encoder = RNNStateEncoder(final_embedding_size, self._hidden_size)
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _initialize_goal_encoder(self, observation_space):
        self._n_input_goal = observation_space.spaces[
            self.goal_sensor_uuid
        ].shape[0]

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def _append_additional_sensors(self, x, observations):
        for sensor in self.additional_sensors:
            x.append(observations[sensor])
        return x

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x.append(perception_embed)
        if self.goal_sensor_uuid is not None:
            x.append(self.get_target_encoding(observations))

        x = self._append_additional_sensors(x, observations)

        x = torch.cat(x, dim=-1) # t x n x -1

        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        return x, rnn_hidden_states
