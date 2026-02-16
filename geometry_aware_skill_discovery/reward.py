# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import torch
import numpy as np
import pickle

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from geometry_aware_skill_discovery.laplacian_metric import LaplacianMetricCalculator
from base.modules.intrinsic_motivation import IntrinsicMotivationModule

class SPECTRAProvider(IntrinsicMotivationModule):
    """
    Provides Topological PBRS rewards using Laplacian Commute Time Distance.
    """
    def __init__(self, maze_type, exp_name="curriculum", laplacian_stage="stage_2",
                 time_penalty=0.0, sparse_bonus=0.0, success_threshold=0.2,
                 gaussian_bonus=0.0, gaussian_std=0.5, reward_scale=1.0):
        super(SPECTRAProvider, self).__init__()
        self.maze_type = maze_type
        self.exp_name = exp_name
        self.time_penalty = float(time_penalty)
        self.sparse_bonus = float(sparse_bonus)
        self.success_threshold = float(success_threshold)
        self.gaussian_bonus = float(gaussian_bonus)
        self.gaussian_std = float(gaussian_std)
        self.reward_scale = float(reward_scale)
        
        # Latest breakdown stats for logging
        self._last_breakdown = {}
        
        # 1. Load Laplacian Calculator
        calc_exp_id = os.path.join(exp_name, laplacian_stage) if exp_name == "curriculum" else exp_name
        self.calc = LaplacianMetricCalculator(maze_type=maze_type, exp_name=calc_exp_id)
        
        # 2. Load Intent Centroids
        centroids_path = os.path.join("logs/spectral_kmeans", maze_type, exp_name, "intent_centroids.pkl")
        if not os.path.exists(centroids_path):
            raise FileNotFoundError("Centroids not found at {0}. Run visualization script first.".format(centroids_path))
            
        with open(centroids_path, 'rb') as f:
            data = pickle.load(f)
            
        self.centroids_psi = torch.from_numpy(data['centroids_psi']).float()
        self.centroids_s = data['centroids_s']
        self.n_skills = data['n_clusters']
        
        print("SPECTRA Reward Provider initialized for {0} (Scale: {1}, Bonus: {2}).".format(
            maze_type, self.reward_scale, self.sparse_bonus))

    def get_goal_for_skill(self, skill_idx):
        if torch.is_tensor(skill_idx):
            return torch.from_numpy(self.centroids_s[skill_idx.cpu().numpy()]).float()
        return self.centroids_s[skill_idx]

    def compute_potential(self, s, skill_idx):
        psi_s = self.calc.transform_space(s, mode="commute")
        if not torch.is_tensor(psi_s):
            psi_s = torch.from_numpy(psi_s).float()
        
        psi_g = self.centroids_psi[skill_idx]
        dist_sq = torch.sum((psi_s - psi_g).pow(2), dim=1)
        return -0.5 * dist_sq * self.reward_scale

    def compute_reward(self, s, s_next, skill_idx, gamma, reward_type="dynamic"):
        """
        Calculates SPECTRA Reward with refinements.
        """
        with torch.no_grad():
            # 1. Potential-based components (Scale-dependent)
            phi_next = self.compute_potential(s_next, skill_idx)
            
            if reward_type == "dynamic":
                phi_curr = self.compute_potential(s, skill_idx)
                pbrs_rew = gamma * phi_next - phi_curr
            else:
                pbrs_rew = phi_next
            
            # 2. Distance-based components (Scale-independent)
            # Calculate raw CTD square directly in psi space
            psi_s_next = self.calc.transform_space(s_next, mode="commute")
            if not torch.is_tensor(psi_s_next):
                psi_s_next = torch.from_numpy(psi_s_next).float().to(s_next.device)
            
            psi_g = self.centroids_psi[skill_idx].to(s_next.device)
            raw_ctd_sq = torch.sum((psi_s_next - psi_g).pow(2), dim=1)
            
            # Constant Time Penalty
            penalty = -torch.ones_like(phi_next) * self.time_penalty
            
            # Gaussian Reward (Using raw CTD directly)
            gauss = torch.zeros_like(phi_next)
            if self.gaussian_bonus > 0:
                gauss = self.gaussian_bonus * torch.exp(-0.5 * raw_ctd_sq / (self.gaussian_std**2))
            
            # Sparse Success Bonus (Physical distance check)
            g_phys = self.get_goal_for_skill(skill_idx).to(s_next.device)
            dist_phys = torch.norm(s_next - g_phys, dim=1)
            
            reached = (dist_phys < self.success_threshold).float()
            bonus = reached * self.sparse_bonus
            
            total_reward = pbrs_rew + penalty + gauss + bonus
            
            # Store breakdown for logging
            self._last_breakdown = {
                'potential': pbrs_rew.mean().item(),
                'penalty': penalty.mean().item(),
                'gauss': gauss.mean().item(),
                'bonus': bonus.mean().item()
            }
            
        return total_reward

    def surprisal(self, batch, gamma=0.99, reward_type="dynamic"):
        """Returns single tensor for compatibility."""
        s = batch['state']
        s_next = batch['next_state']
        skill = batch['skill']
        if skill.dtype != torch.long: skill = skill.long()
        return self.compute_reward(s, s_next, skill, gamma, reward_type=reward_type)