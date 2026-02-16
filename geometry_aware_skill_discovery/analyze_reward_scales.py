# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import torch
import numpy as np
import yaml
from argparse import ArgumentParser

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_aware_skill_discovery.reward import SPECTRAProvider
from agents.maze_agents.toy_maze.env.maze_env import Env

def analyze_scales():
    parser = ArgumentParser(description="Analyze Reward Scales for SPECTRA RL.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--reward_type", type=str, default="dynamic")
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    
    maze_type = config['experiment']['maze_type']
    rl_cfg = config['rl']
    gamma = rl_cfg['common']['gamma']
    
    # 2. Setup Provider
    # Note: reward_scale is important here. We want to see results for BOTH 1.0 and config value.
    provider = SPECTRAProvider(maze_type=maze_type, exp_name=config['experiment']['exp_name'])
    
    # 3. Create Env and Sample Points
    env = Env(n=1, maze_type=maze_type, use_antigoal=False)
    states = torch.stack([torch.tensor(env.sample()) for _ in range(args.num_samples)]).float()
    
    print("
=== SPECTRA Reward Scale Analysis: {0} ===".format(maze_type))
    
    for k in range(min(3, provider.n_skills)): # Check first few skills
        print("
--- Analyzing Skill {0} ---".format(k))
        skill_idx = torch.full((args.num_samples,), k, dtype=torch.long)
        
        # Calculate unscaled potentials (assuming reward_scale=1.0 for base measurement)
        # We temporarily bypass reward_scale to see the raw magnitude
        original_scale = provider.reward_scale
        provider.reward_scale = 1.0
        phis = provider.compute_potential(states, skill_idx)
        
        max_phi = torch.abs(phis).max().item()
        avg_phi = phis.mean().item()
        
        print("Raw Potential Range: [{0:.2f}, 0.00]".format(-max_phi))
        
        # 4. Calculate Passive Gain (the "stay still" profit)
        passive_gain = (1 - gamma) * max_phi
        print("Passive Gain at farthest point (Raw): {0:.4f}".format(passive_gain))
        
        # 5. Calculate Step Delta (moving towards goal)
        # Simulate a small step of 0.1 towards the goal
        g_phys = provider.get_goal_for_skill(k)
        if torch.is_tensor(g_phys): g_phys = g_phys.numpy()
        
        deltas = []
        for i in range(min(100, args.num_samples)):
            s = states[i].numpy()
            direction = g_phys - s
            dist = np.linalg.norm(direction)
            if dist > 0.1:
                step = (direction / dist) * 0.1
                s_next = torch.tensor(s + step).unsqueeze(0).float()
                s_curr = torch.tensor(s).unsqueeze(0).float()
                p_next = provider.compute_potential(s_next, torch.tensor([k]))
                p_curr = provider.compute_potential(s_curr, torch.tensor([k]))
                # PBRS Delta: gamma * Phi_next - Phi_curr
                delta = (gamma * p_next - p_curr).item()
                deltas.append(delta)
        
        avg_delta = np.mean(deltas) if deltas else 0.0
        print("Average PBRS Delta per 0.1 step (Raw): {0:.4f}".format(avg_delta))
        
        # 6. Suggested Scale Recommendation
        # We want avg_delta * scale > time_penalty AND time_penalty > passive_gain * scale
        print("
[Recommendations based on Penalty settings]")
        for test_penalty in [0.01, 0.05, 0.1]:
            # Scale must be small enough so that: passive_gain * scale < test_penalty
            # Scale must be large enough so that: avg_delta * scale > test_penalty
            suggested_scale = test_penalty / (avg_delta + 1e-8)
            print("  For Penalty {0}: Suggested Scale should be around {1:.4f}".format(test_penalty, suggested_scale))
            safe_limit = test_penalty / (passive_gain + 1e-8)
            print("    (Safety check: Scale must be LESS than {0:.4f} to prevent passive gain)".format(safe_limit))

        provider.reward_scale = original_scale

if __name__ == "__main__":
    analyze_scales()
