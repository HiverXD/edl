# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import json
import torch
import numpy as np
import collections
import yaml

# Add project root to path for loading LaplacianEncoder
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from geometry_aware_skill_discovery.train_laplacian_encoder import LaplacianEncoder

class LaplacianMetricCalculator:
    """
    Calculates various Laplacian-based distance metrics and transforms state spaces
    by processing learned eigenvectors and eigenvalues.
    """
    def __init__(self, maze_type, exp_name="default"):
        self.maze_type = maze_type
        self.exp_name = exp_name
        
        # 1. Resolve paths
        root_dir = os.environ.get("ROOT_DIR", ".")
        exp_path = os.path.join(root_dir, "logs/laplacian_encoder", maze_type, exp_name)
        model_path = os.path.join(exp_path, "model.pth.tar")
        stats_path = os.path.join(exp_path, "training_stats.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found at {0}".format(model_path))
        if not os.path.exists(stats_path):
            raise FileNotFoundError("Stats not found at {0}".format(stats_path))
            
        # 2. Load Model & Normalization stats
        checkpoint = torch.load(model_path, map_location='cpu')
        args = checkpoint['args']
        self.mean = torch.from_numpy(checkpoint['mean']).float()
        self.std = torch.from_numpy(checkpoint['std']).float()
        
        self.model = LaplacianEncoder(input_dim=2, hidden_dim=args.hidden_dim, output_dim=args.dim)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        # 3. Load & Sort Eigenvalues
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        # Use eigenvalues from the last epoch
        raw_eigenvalues = np.array(stats[-1]['eigenvalues'])
        
        # Sort eigenvalues in ascending order (smallest to largest)
        self.sort_indices = np.argsort(raw_eigenvalues)
        self.eigenvalues = raw_eigenvalues[self.sort_indices]
        
        # Avoid division by zero for commute time distance
        self.safe_eigenvalues = np.maximum(self.eigenvalues, 1e-6)
        
        print("Loaded Laplacian Metric Calculator for {0}/{1}".format(maze_type, exp_name))
        print("Sorted Eigenvalues: {0}".format(np.round(self.eigenvalues, 4)))

        # 4. Load Metadata (Optional Global Normalization)
        self.global_std_val = 1.0
        meta_path = os.path.join(exp_path, "meta_data.yaml")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = yaml.load(f)
                self.global_std_val = float(meta['normalization'].get('global_psi_std', 1.0))
            print("Applying Global Normalization (Std: {0:.4f})".format(self.global_std_val))

    def encode_sorted(self, states):
        """
        Encodes states and returns eigenvectors sorted by eigenvalue magnitude.
        Now normalized by global_std_val.
        """
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
            
        # Handle single state input
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
            
        # Apply normalization
        states_norm = (states - self.mean.to(states.device)) / self.std.to(states.device)
        phi = self.model(states_norm)
        
        # Reorder dimensions to match sorted eigenvalues
        phi_sorted = phi[:, self.sort_indices]
        
        # Apply Global Scaling
        return phi_sorted / self.global_std_val

    def transform_space(self, states, mode="truncated", **kwargs):
        """
        Transforms coordinates into different Laplacian-weighted vector spaces.
        """
        phi_sorted = self.encode_sorted(states)
        
        # Convert eigenvalues to match device
        device = phi_sorted.device
        safe_ev = torch.from_numpy(self.safe_eigenvalues).float().to(device)
        ev = torch.from_numpy(self.eigenvalues).float().to(device)

        if mode == "truncated":
            return phi_sorted
            
        elif mode == "commute":
            weights = 1.0 / torch.sqrt(safe_ev)
            return phi_sorted * weights
            
        elif mode == "diffusion":
            t = kwargs.get('t', 1.0)
            weights = torch.exp(-ev * t)
            return phi_sorted * weights
            
        else:
            raise ValueError("Unknown mode: {0}".format(mode))

    def calculate_distance(self, s1, s2, mode="diffusion", **kwargs):
        """
        Calculates distance between s1 and s2 using the specified metric.
        """
        phi1 = self.encode_sorted(s1)
        phi2 = self.encode_sorted(s2)
        
        if phi2.shape[0] == 1 and phi1.shape[0] > 1:
            phi2 = phi2.expand(phi1.shape[0], -1)
            
        diff_sq = (phi1 - phi2).pow(2).numpy()
        
        if mode == "truncated":
            n = kwargs.get('n', 2)
            dist_sq = np.sum(diff_sq[:, :n], axis=1)
        elif mode == "diffusion":
            t = kwargs.get('t', 1.0)
            weights = np.exp(-2 * self.eigenvalues * t)
            dist_sq = np.sum(diff_sq * weights, axis=1)
        elif mode == "commute":
            weights = 1.0 / self.safe_eigenvalues
            dist_sq = np.sum(diff_sq * weights, axis=1)
        else:
            raise ValueError("Unknown mode: {0}".format(mode))
            
        return np.sqrt(dist_sq)