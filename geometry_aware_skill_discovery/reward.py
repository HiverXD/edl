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
    def __init__(self, maze_type, exp_name="curriculum", laplacian_stage="stage_2"):
        self.maze_type = maze_type
        self.exp_name = exp_name
        
        # 1. Load Laplacian Calculator
        # Resolve path based on experiment naming conventions
        calc_exp_id = os.path.join(exp_name, laplacian_stage) if exp_name == "curriculum" else exp_name
        self.calc = LaplacianMetricCalculator(maze_type=maze_type, exp_name=calc_exp_id)
        
        # 2. Load Intent Centroids
        centroids_path = os.path.join("logs/spectral_kmeans", maze_type, exp_name, "intent_centroids.pkl")
        if not os.path.exists(centroids_path):
            raise FileNotFoundError("Centroids not found at {0}. Run visualization script first.".format(centroids_path))
            
        with open(centroids_path, 'rb') as f:
            data = pickle.load(f)
            
        # These are in the scaled Laplacian (Commute) space
        self.centroids_psi = torch.from_numpy(data['centroids_psi']).float()
        self.centroids_s = data['centroids_s'] # Physical coordinates for reference
        self.n_skills = data['n_clusters']
        
        print("SPECTRA Reward Provider initialized for {0}.".format(maze_type))
        print("Loaded {0} intent centroids.".format(self.n_skills))

    def get_goal_for_skill(self, skill_idx):
        """Returns the physical coordinates of the target centroid."""
        # Handle batch or single index
        if torch.is_tensor(skill_idx):
            return torch.from_numpy(self.centroids_s[skill_idx.cpu().numpy()]).float()
        return self.centroids_s[skill_idx]

    def compute_potential(self, s, skill_idx):
        """
        Calculates Phi(s, g) = -0.5 * ||psi(s) - psi(g)||^2
        Note: psi is the commute-scaled embedding.
        """
        # 1. Transform s to psi space (Weighted by 1/sqrt(lambda))
        # Important: Use the same transformation as K-means!
        psi_s = self.calc.transform_space(s, mode="commute") # returns numpy or tensor
        if not torch.is_tensor(psi_s):
            psi_s = torch.from_numpy(psi_s).float()
        
        # 2. Get target centroids psi
        # skill_idx is expected to be a tensor of indices (N,)
        psi_g = self.centroids_psi[skill_idx] # (N, D)
        
        # 3. Calculate squared distance: ||psi(s) - psi(g)||^2
        dist_sq = torch.sum((psi_s - psi_g).pow(2), dim=1)
        
        # 4. Return Potential
        return -0.5 * dist_sq

    def compute_reward(self, s, s_next, skill_idx, gamma):
        """
        Calculates SPECTRA PBRS Reward: R = gamma * Phi(s', g) - Phi(s, g)
        """
        with torch.no_grad():
            phi_next = self.compute_potential(s_next, skill_idx)
            phi_curr = self.compute_potential(s, skill_idx)
            
            reward = gamma * phi_next - phi_curr
            
        return reward

    def surprisal(self, batch, gamma=0.99):
        """
        Interface compatible with existing EDL Learners.
        Calculates PBRS reward for a batch of transitions.
        """
        # Expects batch to have 'state', 'next_state', and 'skill'
        s = batch['state']
        s_next = batch['next_state']
        skill = batch['skill']
        
        # Ensure skill is in index form (Long)
        if skill.dtype != torch.long:
            # If it's continuous (interpolated), find nearest neighbor or handle accordingly
            # For now, we assume discrete skill indices as per Phase 2 K-means
            skill = skill.long()
            
        return self.compute_reward(s, s_next, skill, gamma)
