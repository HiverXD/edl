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

from geometry_aware_skill_discovery.reward import SPECTRAProvider

def verify_sparse_reward():
    parser = ArgumentParser(description="Verify if Sparse Rewards are actually being generated.")
    parser.add_argument("--maze_type", type=str, default="square_corridor")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f: config = yaml.load(f)
    common = config['rl']['common']
    static = config['rl']['static']
    
    # 2. Setup Provider with ACTUAL config values
    provider = SPECTRAProvider(
        maze_type=args.maze_type, 
        exp_name=config['experiment']['exp_name'],
        sparse_bonus=common.get('sparse_bonus', 0.0),
        success_threshold=common.get('success_threshold', 0.15),
        gaussian_bonus=common.get('gaussian_bonus', 0.0),
        gaussian_std=common.get('gaussian_std', 0.5),
        reward_scale=static.get('reward_scale', 1.0)
    )
    
    threshold = provider.success_threshold
    bonus = provider.sparse_bonus
    
    print("\n--- SPECTRA Sparse Reward Integrity Check ---")
    print("Loaded from {0}:".format(args.config))
    print("  Threshold: {0:.4f}".format(threshold))
    print("  Bonus:     {0:.4f}".format(bonus))
    print("  Scale:     {0:.4f}".format(provider.reward_scale))
    
    for skill_idx in range(3):
        print("\n[Skill {0}]".format(skill_idx))
        goal_np = provider.get_goal_for_skill(skill_idx)
        if torch.is_tensor(goal_np): goal_np = goal_np.numpy()
        
        # Test Point A: Exactly at Goal
        s_next_a = torch.from_numpy(goal_np).unsqueeze(0).float()
        s_curr = s_next_a.clone()
        
        rew_a = provider.compute_reward(s_curr, s_next_a, torch.tensor([skill_idx]), gamma=0.99)
        breakdown_a = provider._last_breakdown
        
        print("  At Goal (Dist 0.00): Total Rew={0:.4f}, Bonus={1:.4f}".format(rew_a.item(), breakdown_a['bonus']))
        
        # Test Point B: Just Inside Threshold
        offset = np.array([[threshold - 0.01, 0.0]])
        s_next_b = torch.from_numpy(goal_np + offset).float()
        rew_b = provider.compute_reward(s_curr, s_next_b, torch.tensor([skill_idx]), gamma=0.99)
        breakdown_b = provider._last_breakdown
        print("  Inside Threshold (Dist {0:.3f}): Total Rew={1:.4f}, Bonus={2:.4f}".format(threshold-0.01, rew_b.item(), breakdown_b['bonus']))

        # Test Point C: Just Outside Threshold
        offset = np.array([[threshold + 0.01, 0.0]])
        s_next_c = torch.from_numpy(goal_np + offset).float()
        rew_b = provider.compute_reward(s_curr, s_next_c, torch.tensor([skill_idx]), gamma=0.99)
        breakdown_c = provider._last_breakdown
        print("  Outside Threshold (Dist {0:.3f}): Total Rew={1:.4f}, Bonus={2:.4f}".format(threshold+0.01, rew_b.item(), breakdown_c['bonus']))

    print("\n--- Final Verdict ---")
    if bonus > 0 and breakdown_a['bonus'] > 0:
        print("[SUCCESS] Reward engine IS correctly assigning bonuses.")
    elif bonus > 0:
        print("[CRITICAL FAIL] Bonus is set to {0} but engine outputs 0.0!".format(bonus))
    else:
        print("[WARNING] Bonus is set to 0.0 in config. No sparse signal will be given.")

if __name__ == "__main__":
    verify_sparse_reward()