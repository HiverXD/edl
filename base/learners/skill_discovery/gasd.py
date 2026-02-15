# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import torch
import json
from .edl import BaseEDLLearner
from ..sac_v2 import BaseSACV2Learner
from geometry_aware_skill_discovery.reward import SPECTRAProvider
from agents.maze_agents.toy_maze.env import Env
from agents.maze_agents.modules import StochasticPolicy
from agents.maze_agents.modules.value_function import Critic

class GASDSACV2Learner(BaseSACV2Learner, BaseEDLLearner):
    """
    Geometry Aware Skill Discovery (GASD) Learner.
    Uses SAC-v2 for policy optimization and SPECTRA (Laplacian PBRS) for rewards.
    """
    AGENT_TYPE = 'GASD'

    def __init__(self, **kwargs):
        # 1. Extract and store GASD specific parameters
        self.maze_type = kwargs.pop('maze_type', 'spiral')
        self.ep_len = kwargs.pop('n', 50)
        self.exp_name = kwargs.pop('exp_name', 'curriculum')
        self.pbrs_gamma = float(kwargs.pop('pbrs_gamma', 0.99))
        self.reward_type = str(kwargs.pop('reward_type', 'dynamic')).lower()
        
        # New reward refinements
        self.time_penalty = float(kwargs.pop('time_penalty', 0.0))
        self.sparse_bonus = float(kwargs.pop('sparse_bonus', 0.0))
        self.success_threshold = float(kwargs.pop('success_threshold', 0.2))
        self.gaussian_bonus = float(kwargs.pop('gaussian_bonus', 0.0))
        self.gaussian_std = float(kwargs.pop('gaussian_std', 0.5))
        self.reward_scale = float(kwargs.pop('reward_scale', 1.0))
        
        self.im_nu_val = float(kwargs.get('im_nu', 1.0))
        if 'im_params' not in kwargs:
            kwargs['im_params'] = {'nu': self.im_nu_val, 'type': 'SPECTRA'}
        
        # 2. Base Initializations (calls _make_im_modules)
        super().__init__(**kwargs)
        
        # 3. Post-init cleanup
        self.im_nu = self.im_nu_val
        self.im_lambda = 0.0
        
        print("GASD Learner initialized with SAC-v2 and SPECTRA {0} Rewards (Scale: {1}).".format(
            self.reward_type, self.reward_scale))

    def _init_spectra_internal(self):
        """Standardized initialization for the SPECTRA provider."""
        self.im = SPECTRAProvider(
            maze_type=self.maze_type, 
            exp_name=self.exp_name,
            time_penalty=self.time_penalty,
            sparse_bonus=self.sparse_bonus,
            success_threshold=self.success_threshold,
            gaussian_bonus=self.gaussian_bonus,
            gaussian_std=self.gaussian_std,
            reward_scale=self.reward_scale
        )
        self.skill_dim = self.im.n_skills

    def create_env(self):
        from agents.maze_agents.toy_maze.env.maze_env import Env
        params = self.env_params.copy()
        params['maze_type'] = self.maze_type
        params['n'] = self.ep_len
        return Env(**params)

    def _make_im_modules(self):
        self._init_spectra_internal()
        return self.im

    def _make_agent_modules(self):
        if not hasattr(self, 'skill_dim'):
            self._init_spectra_internal()
            
        skill_n = self.skill_dim
        self.skill_embedding = torch.nn.Embedding(skill_n, skill_n)
        self.skill_embedding.weight.data.copy_(torch.eye(skill_n))
        self.skill_embedding.weight.requires_grad = False
        
        common_kwargs = dict(
            env=self._dummy_env,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            goal_size=skill_n,
            normalize_inputs=self.normalize_inputs
        )
        
        self.policy = StochasticPolicy(**common_kwargs)
        self.q1 = Critic(**common_kwargs)
        self.q2 = Critic(**common_kwargs)
        self.q1_target = Critic(**common_kwargs)
        self.q2_target = Critic(**common_kwargs)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def _make_agent(self):
        from agents.maze_agents.toy_maze.skill_discovery.edl import DistanceStochasticAgent
        
        if self.im is None:
            self._init_spectra_internal()
            
        class DummyVAE:
            def __init__(self, provider): self.provider = provider
            def get_centroids(self, batch):
                return self.provider.get_goal_for_skill(batch['skill'])
        
        vae_bridge = DummyVAE(self.im)
        
        return DistanceStochasticAgent(
            env=self.create_env(), 
            policy=self.policy, 
            skill_n=self.skill_dim,
            skill_embedding=self.skill_embedding, 
            vae=vae_bridge
        )

    def relabel_batch(self, batch):
        with torch.no_grad():
            spectra_rew = self.im.surprisal(batch, gamma=self.pbrs_gamma, reward_type=self.reward_type)
            
        im_nu = self.im_nu if self.im_nu is not None else 1.0
        batch['reward'] = (batch.get('env_reward', 0.0) * float(self.env_reward)) + (im_nu * spectra_rew)
        batch['im_reward'] = spectra_rew
        
        return batch

    def get_optim_params(self):
        params = super().get_optim_params()
        if hasattr(self, 'log_alpha'):
            base_params = params[0]['params']
            if not any(p is self.log_alpha for p in base_params):
                base_params.append(self.log_alpha)
        return params

    def get_aux_optim_params(self):
        return []

    def get_im_loss(self, batch):
        return torch.tensor(0.0).to(batch['state'].device)

    def sample_skill(self):
        return torch.randint(0, self.skill_dim, (1,))

    def preprocess_skill(self, z):
        if z.dtype != torch.long: z = z.long()
        return self.skill_embedding(z)

    def get_values(self, batch): return torch.zeros_like(batch['reward'])
    def get_terminal_values(self, batch): return torch.zeros_like(batch['reward'][-1:])