# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import torch
import json
import numpy as np
import pickle
from .edl import BaseEDLLearner
from ..sac_v2 import BaseSACV2Learner
from geometry_aware_skill_discovery.reward import SPECTRAProvider
from agents.maze_agents.toy_maze.env import Env
from agents.maze_agents.modules import StochasticPolicy
from agents.maze_agents.modules.value_function import Critic

class GASDSACV2Learner(BaseSACV2Learner, BaseEDLLearner):
    """
    Geometry Aware Skill Discovery (GASD) Learner.
    Final Fixed Version: Corrects reward relabeling and logging indices.
    """
    AGENT_TYPE = 'GASD'

    def __init__(self, **kwargs):
        # 1. Extract GASD parameters
        self.maze_type = kwargs.pop('maze_type', 'spiral')
        self.ep_len = kwargs.pop('n', 50)
        self.exp_name = kwargs.pop('exp_name', 'curriculum')
        self.pbrs_gamma = float(kwargs.pop('pbrs_gamma', 0.99))
        self.reward_type = str(kwargs.pop('reward_type', 'dynamic')).lower()
        
        self.time_penalty = float(kwargs.pop('time_penalty', 0.0))
        self.sparse_bonus = float(kwargs.pop('sparse_bonus', 0.0))
        self.success_threshold = float(kwargs.pop('success_threshold', 0.15))
        self.gaussian_bonus = float(kwargs.pop('gaussian_bonus', 0.0))
        self.gaussian_std = float(kwargs.pop('gaussian_std', 0.5))
        self.reward_scale = float(kwargs.pop('reward_scale', 1.0))
        self.im_nu_val = float(kwargs.get('im_nu', 1.0))
        
        self.master_keys = kwargs.pop('logging_keys', [])
        if not self.master_keys:
            self.master_keys = ["success", "dist_to_goal", "cum_rew_total", "cum_rew_im", "cum_rew_ext", 
                                "batch_rew_mean", "avg_q1", "avg_q2", "q1_loss", "q2_loss", 
                                "p_loss", "alpha_loss", "alpha", "rew_pot", "rew_pen", "rew_gauss", "rew_bon"]

        if 'im_params' not in kwargs:
            kwargs['im_params'] = {'nu': self.im_nu_val, 'type': 'SPECTRA'}
        
        super(GASDSACV2Learner, self).__init__(**kwargs)
        
        self.ep_summary_keys = self.master_keys
        self.im_nu = self.im_nu_val
        self.im_lambda = 0.0
        
        # Local stats store to ensure fill_summary gets the RIGHT data
        self._current_ep_breakdown = {}

    def _make_im_modules(self):
        self.im = SPECTRAProvider(
            maze_type=self.maze_type, exp_name=self.exp_name,
            time_penalty=self.time_penalty, sparse_bonus=self.sparse_bonus,
            success_threshold=self.success_threshold, gaussian_bonus=self.gaussian_bonus,
            gaussian_std=self.gaussian_std, reward_scale=self.reward_scale
        )
        self.skill_dim = self.im.n_skills
        return self.im

    def relabel_episode(self):
        """
        Manually relabel the current episode with SPECTRA rewards and record distances.
        """
        self._compress_me = []
        ep = self.agent.episode
        self.distance = [] # Reset for current ep
        
        # Convert episode to batch for vector processing
        batched = {key: torch.stack([e[key] for e in ep]) for key in ep[0].keys()}
        
        with torch.no_grad():
            total_rewards = self.im.surprisal(batched, gamma=self.pbrs_gamma, reward_type=self.reward_type)
            breakdown = self.im._last_breakdown
            
        # Update each transition and record distances
        for i, e in enumerate(ep):
            spectra_r = total_rewards[i].item()
            e['reward'] = torch.tensor(spectra_r * self.im_nu)
            e['im_reward'] = torch.tensor(spectra_r)
            e['env_reward'] = torch.tensor(0.0)
            
            # Record distance to goal for logging
            d = self._dummy_env.dist(e['next_state'], e['goal']).item()
            self.distance.append(d)
            
        # Capture breakdown mean for this specific episode summary
        self._current_ep_breakdown = breakdown
        self._compress_me.append(ep)

    def fill_summary(self, *values):
        """
        Final authority on logging order.
        Values from SAC: (r_mean, q1, q2, q1_l, q2_l, p_l, a_l, alpha)
        """
        b = self._current_ep_breakdown
        ep = self.agent.episode
        
        summary = [
            float(self.was_success),
            float(self.dist_to_goal),
            float(sum([e['reward'].item() for e in ep])),
            float(sum([e['im_reward'].item() for e in ep])),
            0.0, # cum_rew_ext
            
            # SAC internal stats
            values[0].item(), values[1].item(), values[2].item(),
            values[3].item(), values[4].item(), values[5].item(),
            values[6].item(), values[7].item(),
            
            # SPECTRA detailed components
            b.get('potential', 0.0),
            b.get('penalty', 0.0),
            b.get('gauss', 0.0),
            b.get('bonus', 0.0)
        ]
        self._ep_summary = summary

    def _make_agent_modules(self):
        if not hasattr(self, 'skill_dim') or self.im is None: self._make_im_modules()
        skill_n = self.skill_dim
        self.skill_embedding = torch.nn.Embedding(skill_n, self.im.centroids_psi.shape[1])
        self.skill_embedding.weight.data.copy_(self.im.centroids_psi); self.skill_embedding.weight.requires_grad = False
        kwargs = dict(env=self._dummy_env, hidden_size=self.hidden_size, num_layers=self.num_layers,
                      goal_size=self.skill_embedding.embedding_dim, normalize_inputs=self.normalize_inputs)
        self.policy = StochasticPolicy(**kwargs)
        self.q1 = Critic(**kwargs); self.q2 = Critic(**kwargs)
        self.q1_target = Critic(**kwargs); self.q2_target = Critic(**kwargs)
        self.q1_target.load_state_dict(self.q1.state_dict()); self.q2_target.load_state_dict(self.q2.state_dict())

    def create_env(self):
        from agents.maze_agents.toy_maze.env.maze_env import Env
        params = self.env_params.copy()
        params['maze_type'] = self.maze_type; params['n'] = self.ep_len; params['done_on_success'] = True
        return Env(**params)

    def _make_agent(self):
        from agents.maze_agents.toy_maze.skill_discovery.edl import DistanceStochasticAgent
        class DummyVAE:
            def __init__(self, p): self.p = p
            def get_centroids(self, b): return self.p.get_goal_for_skill(b['skill'])
        return DistanceStochasticAgent(env=self.create_env(), policy=self.policy, skill_n=self.skill_dim,
                                       skill_embedding=self.skill_embedding, vae=DummyVAE(self.im))

    def relabel_batch(self, batch):
        with torch.no_grad():
            spectra_rew = self.im.surprisal(batch, gamma=self.pbrs_gamma, reward_type=self.reward_type)
        batch['reward'] = (batch.get('env_reward', 0.0) * float(self.env_reward)) + (self.im_nu * spectra_rew)
        batch['im_reward'] = spectra_rew
        return batch

    def get_optim_params(self):
        params = super(GASDSACV2Learner, self).get_optim_params()
        if hasattr(self, 'log_alpha'):
            base_params = params[0]['params']
            if not any(p is self.log_alpha for p in base_params): base_params.append(self.log_alpha)
        return params

    def get_aux_optim_params(self): return []
    def get_im_loss(self, batch): return torch.tensor(0.0).to(batch['state'].device)
    def sample_skill(self): return torch.randint(0, self.skill_dim, (1,))
    def preprocess_skill(self, z):
        if z.dtype != torch.long: z = z.long()
        return self.skill_embedding(z)
    def get_values(self, batch): return torch.zeros_like(batch['reward'])
    def get_terminal_values(self, batch): return torch.zeros_like(batch['reward'][-1:])
