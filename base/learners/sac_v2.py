# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
from .base import BaseLearner
from agents.maze_agents.modules import StochasticPolicy, Value # Actually we need Critic/Q modules

class BaseSACV2Learner(BaseLearner):
    """
    Base Learner for modern SAC-v2.
    Defines Twin Q networks and handles target network synchronization.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _make_agent_modules(self):
        """
        Initializes networks required for SAC-v2.
        Note: We reuse StochasticPolicy and create Twin Q functions.
        """
        # 1. Policy Network
        # Goal size corresponds to code_size of VAE or Skill dimension
        goal_size = self.vae.code_size if hasattr(self, 'vae') else self.skill_dim
        
        policy_kwargs = dict(
            env=self._dummy_env, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers,
            goal_size=goal_size, 
            normalize_inputs=self.normalize_inputs
        )
        
        self.policy = StochasticPolicy(**policy_kwargs)

        # 2. Twin Q-Networks
        # Use the 'Critic' module which takes (state, action, goal)
        from agents.maze_agents.modules.value_function import Critic
        
        q_kwargs = dict(
            env=self._dummy_env, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers,
            goal_size=goal_size, 
            normalize_inputs=self.normalize_inputs,
            use_antigoal=getattr(self, 'use_antigoal', False)
        )
        
        # Twin Q (Standard SAC Twin-Critic setup)
        self.q1 = Critic(**q_kwargs)
        self.q2 = Critic(**q_kwargs)
        
        # Target Q (Delayed updates for stability)
        self.q1_target = Critic(**q_kwargs)
        self.q2_target = Critic(**q_kwargs)
        
        # Initialize targets with current weights
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def soft_update(self):
        """
        Polyak update for target networks: target = target * polyak + current * (1 - polyak)
        Note: Our sac_v2_decorator uses polyak=0.995 (standard v2 style).
        """
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(target_param.data * self.polyak + param.data * (1.0 - self.polyak))
            
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(target_param.data * self.polyak + param.data * (1.0 - self.polyak))

    def get_curr_qs(self, batch, new_actions=None, q_i=1):
        """
        Computes Q_i(s, a). 
        If new_actions is provided, uses them instead of batch['action'].
        """
        action = new_actions if new_actions is not None else batch['action']
        q_module = self.q1 if q_i == 1 else self.q2
        
        return q_module(
            batch['state'],
            action,
            self.preprocess_skill(batch['skill'])
        )

    def get_action_qs(self, batch, q_i=1):
        """ Alias for compatibility with decorators """
        return self.get_curr_qs(batch, q_i=q_i)

    def get_next_qs(self, batch, new_actions=None, q_i=1):
        """
        Computes Q_target_i(s', a'). 
        Used for Bellman target calculation.
        """
        # Note: new_actions (a') must be sampled from current policy at next_state
        q_target_module = self.q1_target if q_i == 1 else self.q2_target
        
        return q_target_module(
            batch['next_state'],
            new_actions,
            self.preprocess_skill(batch['skill'])
        )
