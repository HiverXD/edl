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

    def __init__(self, maze_type, exp_name="curriculum", pbrs_gamma=0.99, **kwargs):
        # 1. Initialize SPECTRA Reward Provider
        # This loads the Laplacian model and K-means centroids
        self.spectra = SPECTRAProvider(maze_type=maze_type, exp_name=exp_name)
        self.pbrs_gamma = float(pbrs_gamma)
        
        # Skill dimension is determined by the number of clusters in K-means
        self.skill_dim = self.spectra.n_skills
        
        # 2. Base Initializations
        # Note: BaseEDLLearner handles many bookkeeping keys
        super().__init__(maze_type=maze_type, **kwargs)
        
        print("GASD Learner initialized with SAC-v2 and SPECTRA Rewards.")

    def _make_agent_modules(self):
        """
        Overrides to set up networks for GASD.
        We don't need a VAE discriminator here as rewards are from SPECTRAProvider.
        """
        # Skill size is the number of discrete intents (one-hot or index)
        # However, for the policy/critic, we use the skill embedding if we want continuous, 
        # but here we use simple indices or one-hots. 
        # For compatibility with existing policy modules, we'll treat skill_n as codebook_size.
        
        skill_n = self.skill_dim
        # We'll use a simple identity embedding for discrete skills
        self.skill_embedding = torch.nn.Embedding(skill_n, skill_n)
        self.skill_embedding.weight.data.copy_(torch.eye(skill_n))
        self.skill_embedding.weight.requires_grad = False
        
        # Policy & Twin Q (Using dimensions from Env and SPECTRA)
        # Goal size = skill_n (one-hot representation)
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
        """Creates the agent that will interact with the environment."""
        from agents.maze_agents.toy_maze.skill_discovery.edl import DistanceStochasticAgent
        
        # We need a dummy VAE object to satisfy the agent's expectation of centroid lookup
        class DummyVAE:
            def __init__(self, spectra): self.spectra = spectra
            def get_centroids(self, batch):
                return self.spectra.get_goal_for_skill(batch['skill'])
        
        vae_bridge = DummyVAE(self.spectra)
        
        return DistanceStochasticAgent(
            env=self.create_env(), 
            policy=self.policy, 
            skill_n=self.skill_dim,
            skill_embedding=self.skill_embedding, 
            vae=vae_bridge
        )

    def relabel_batch(self, batch):
        """
        Core of SPECTRA: Replace VAE log-prob with Topological PBRS reward.
        """
        with torch.no_grad():
            # Calculate SPECTRA reward: gamma * Phi(s') - Phi(s)
            spectra_rew = self.spectra.surprisal(batch, gamma=self.pbrs_gamma)
            
        # Update batch rewards
        # extrinsic + intrinsic (spectra)
        im_nu = self.im_nu if self.im_nu is not None else 1.0
        
        # batch['reward'] might already contain env_reward
        batch['reward'] = (batch.get('env_reward', 0.0) * float(self.env_reward)) + (im_nu * spectra_rew)
        batch['im_reward'] = spectra_rew
        
        return batch

    def sample_skill(self):
        """Samples a random skill index."""
        return torch.randint(0, self.skill_dim, (1,))

    def preprocess_skill(self, z):
        """Converts skill index to embedding (one-hot)."""
        if z.dtype != torch.long:
            z = z.long()
        return self.skill_embedding(z)

    # V-function methods are not needed for SAC-v2 but kept for interface compatibility
    def get_values(self, batch): return torch.zeros_like(batch['reward'])
    def get_terminal_values(self, batch): return torch.zeros_like(batch['reward'][-1:])
