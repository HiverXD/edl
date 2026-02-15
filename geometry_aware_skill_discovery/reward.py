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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_aware_skill_discovery.laplacian_metric import LaplacianMetricCalculator

class SPECTRAProvider:
    """
    Provides Topological PBRS rewards using Laplacian Commute Time Distance.
    Matches the objective: R(s, s', g) = gamma * Phi(s', g) - Phi(s, g)
    where Phi(s, g) = -0.5 * CTD(s, g)^2
    """
    def __init__(self, maze_type, exp_name="curriculum", laplacian_stage="stage_2",
                 time_penalty=0.0, sparse_bonus=0.0, success_threshold=0.2):
        self.maze_type = maze_type
        self.exp_name = exp_name
        self.time_penalty = float(time_penalty)
        self.sparse_bonus = float(sparse_bonus)
        self.success_threshold = float(success_threshold)
        
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
        
        print("SPECTRA Reward Provider initialized for {0} (Penalty: {1}, Bonus: {2}).".format(
            maze_type, self.time_penalty, self.sparse_bonus))

    def get_goal_for_skill(self, skill_idx):
        """Returns the physical coordinates of the target centroid."""
        if torch.is_tensor(skill_idx):
            return torch.from_numpy(self.centroids_s[skill_idx.cpu().numpy()]).float()
        return self.centroids_s[skill_idx]

    def compute_potential(self, s, skill_idx):
        """Calculates Phi(s, g) = -0.5 * ||psi(s) - psi(g)||^2"""
        psi_s = self.calc.transform_space(s, mode="commute")
        if not torch.is_tensor(psi_s):
            psi_s = torch.from_numpy(psi_s).float()
        
        psi_g = self.centroids_psi[skill_idx]
        dist_sq = torch.sum((psi_s - psi_g).pow(2), dim=1)
        return -0.5 * dist_sq

    def compute_reward(self, s, s_next, skill_idx, gamma, reward_type="dynamic"):
        """Calculates SPECTRA Reward with refinements."""
        with torch.no_grad():
            phi_next = self.compute_potential(s_next, skill_idx)
            
            if reward_type == "dynamic":
                phi_curr = self.compute_potential(s, skill_idx)
                reward = gamma * phi_next - phi_curr
            else:
                reward = phi_next
            
            # 2. Add Constant Time Penalty
            reward -= self.time_penalty
            
            # 3. Add Sparse Success Bonus
            ctd_next = torch.sqrt(torch.abs(phi_next) * 2.0)
            reached = (ctd_next < self.success_threshold).float()
            reward += reached * self.sparse_bonus
            
        return reward

    def surprisal(self, batch, gamma=0.99, reward_type="dynamic"):
        """
        Interface compatible with existing EDL Learners.
        """
        s = batch['state']
        s_next = batch['next_state']
        skill = batch['skill']
        
        if skill.dtype != torch.long:
            skill = skill.long()
            
        return self.compute_reward(s, s_next, skill, gamma, reward_type=reward_type)
