# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import yaml
import pickle
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_aware_skill_discovery.laplacian_metric import LaplacianMetricCalculator
from agents.maze_agents.toy_maze.env.maze_env import Env

def main():
    parser = ArgumentParser(description="Prepare GASD Discovery Data by sampling the environment and transforming states.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to curriculum config file")
    parser.add_argument("--mode", type=str, default="commute", choices=["identity", "commute", "diffusion"], 
                        help="Transformation mode for states")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of continuous samples to draw from Env")
    parser.add_argument("--use_buffer", action="store_true", help="If set, use transitions.pkl instead of random sampling")
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    exp_cfg = config['experiment']
    vq_cfg = config.get('vqvae', {})
    maze_type = exp_cfg['maze_type']
    exp_name = exp_cfg['exp_name']
    
    # Prioritize mode and num_samples from config, fallback to CLI argument
    mode = vq_cfg.get('mode', args.mode)
    num_samples = vq_cfg.get('num_samples', args.num_samples)
    
    print("--- Preparing GASD Data (Mode: {0}) for {1} ---".format(mode, maze_type))

    save_dir = os.path.join("data/oracle_transitions", exp_name, maze_type, mode)
    save_path = os.path.join(save_dir, "vqvae_oracle.pkl")

    # 2. Collect State Points
    if args.use_buffer:
        # Legacy/Buffer mode: Reuse points from transitions.pkl
        if exp_name == "curriculum":
            source_path = os.path.join("data/oracle_transitions", exp_name, maze_type, "stage_2", maze_type, "transitions.pkl")
        else:
            source_path = os.path.join("data/oracle_transitions", exp_name, maze_type, "transitions.pkl")
            
        print("Loading states from buffer: {0}".format(source_path))
        with open(source_path, 'rb') as f:
            data = pickle.load(f)
        transitions = data['raw_transitions']
        all_states = np.array([t[0] for t in transitions] + [t[2] for t in transitions])
    else:
        # Discovery mode: Continuous uniform sampling (Matching original EDL)
        print("Sampling {0} continuous points from environment...".format(num_samples))
        env = Env(n=1, maze_type=maze_type, use_antigoal=False)
        all_states = np.zeros((num_samples, 2))
        for i in tqdm(range(num_samples), desc="Sampling Env"):
            all_states[i] = env.sample()

    print("Total state points collected: {0}".format(len(all_states)))

    # 3. Transform States
    if mode == "identity":
        transformed_data = all_states
    else:
        # Resolve which model to use for transformation
        calc_exp_name = os.path.join(exp_name, "stage_2") if exp_name == "curriculum" else exp_name
        calc = LaplacianMetricCalculator(maze_type=maze_type, exp_name=calc_exp_name)
        
        print("Transforming states to {0} space...".format(mode))
        batch_size = 10000
        transformed_list = []
        for i in range(0, len(all_states), batch_size):
            batch = all_states[i : i + batch_size]
            transformed_batch = calc.transform_space(batch, mode=mode)
            transformed_list.append(transformed_batch)
        transformed_data = np.concatenate(transformed_list, axis=0)

    # 4. Save Processed Data
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    output = {
        'dataset': torch.from_numpy(transformed_data).float(),
        'mode': mode,
        'maze_type': maze_type,
        'exp_name': exp_name
    }

    with open(save_path, 'wb') as f:
        pickle.dump(output, f)

    print("[Done] Processed data saved to {0}".format(save_path))
    print("Shape: {0}".format(transformed_data.shape))

if __name__ == "__main__":
    main()