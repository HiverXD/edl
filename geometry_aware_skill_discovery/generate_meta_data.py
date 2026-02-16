# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import torch
import numpy as np
import yaml
import pickle
from argparse import ArgumentParser

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_aware_skill_discovery.laplacian_metric import LaplacianMetricCalculator

def generate_meta_data():
    parser = ArgumentParser(description="Calculate Global Std for Laplacian Normalization.")
    parser.add_argument("--maze_type", type=str, default="square_corridor")
    parser.add_argument("--exp_name", type=str, default="curriculum/stage_2")
    args = parser.parse_args()

    # 1. Setup Calculator (without normalization for now)
    calc = LaplacianMetricCalculator(maze_type=args.maze_type, exp_name=args.exp_name)
    
    # 2. Load Sample States to calculate Std
    oracle_path = os.path.join("data/oracle_transitions", args.exp_name, args.maze_type, "transitions.pkl")
    if not os.path.exists(oracle_path):
        print("Oracle not found at {0}. Sampling random points...".format(oracle_path))
        from agents.maze_agents.toy_maze.env.maze_env import Env
        env = Env(n=1, maze_type=args.maze_type)
        samples = np.stack([env.sample() for _ in range(10000)])
    else:
        with open(oracle_path, 'rb') as f:
            data = pickle.load(f)
        samples = data['s']
    
    # 3. Compute Embeddings
    print("Computing embeddings for {0} samples...".format(len(samples)))
    psi = calc.transform_space(samples, mode="commute") # This is now a Tensor due to previous edit
    
    # 4. Calculate Global Scalar Std
    # Ensure it's a numpy array for np.std
    if torch.is_tensor(psi):
        psi_np = psi.detach().cpu().numpy()
    else:
        psi_np = psi
        
    global_std = float(np.std(psi_np))
    
    print("Detected Global Std: {0:.4f}".format(global_std))
    
    # 5. Save to meta_data.yaml
    meta_dir = os.path.join("logs/laplacian_encoder", args.maze_type, args.exp_name)
    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)
    meta_path = os.path.join(meta_dir, "meta_data.yaml")
    
    meta_content = {
        'normalization': {
            'global_psi_std': global_std,
            'mode': 'global_scalar'
        }
    }
    
    with open(meta_path, 'w') as f:
        yaml.dump(meta_content, f)
        
    print("Saved metadata to: {0}".format(meta_path))

if __name__ == "__main__":
    generate_meta_data()