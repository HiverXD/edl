# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from argparse import ArgumentParser

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_aware_skill_discovery.reward import SPECTRAProvider
from agents.maze_agents.toy_maze.env.maze_env import Env
from agents import agent_classes
from result_inspection.toy_maze import ENV_LIMS

def run_optimistic_q_gallery():
    parser = ArgumentParser(description="Visualize Optimistic Q-Value Gallery (Max Q over 8 directions).")
    parser.add_argument("--maze_type", type=str, default="square_corridor")
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    # 1. Setup Provider and Env
    with open('config.yaml', 'r') as f: config = yaml.load(f)
    provider = SPECTRAProvider(maze_type=args.maze_type, exp_name=config['experiment']['exp_name'])
    env = Env(n=1, maze_type=args.maze_type)
    
    # 2. Load Policy/Critic
    AgentClass = agent_classes('maze', 'GASD', 'SAC_V2')
    agent_params = config['rl']['common'].copy()
    agent_params.update(config['rl']['static'])
    agent_params.update({
        'maze_type': args.maze_type, 'exp_name': config['experiment']['exp_name'], 
        'reward_type': 'static', 'logging_keys': config['rl']['common'].get('logging_keys', [])
    })
    
    learner = AgentClass(**agent_params)
    learner.load_checkpoint(args.model_path)
    learner.eval()

    # 3. Generate Grid and Action Samples
    lims = ENV_LIMS[args.maze_type]
    res_x, res_y = 40, 10
    x = np.linspace(lims['x'][0], lims['x'][1], res_x)
    y = np.linspace(lims['y'][0], lims['y'][1], res_y)
    X, Y = np.meshgrid(x, y)
    
    # 8-Direction actions (Unit vectors)
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    test_actions = torch.tensor([[np.cos(a), np.sin(a)] for a in angles]).float()
    # Scale by action range
    test_actions = test_actions * env.action_range

    # 4. Find Global Range for Colorbar (Optimistic version)
    print("\n--- Calculating Global Optimistic Q Range ---")
    all_q_data = []
    for skill_idx in range(10):
        skill_vec = learner.preprocess_skill(torch.tensor([skill_idx]))
        for i in range(res_y):
            for j in range(res_x):
                if not env.maze.is_inside_wall((X[i, j], Y[i, j])):
                    s_torch = torch.tensor([X[i, j], Y[i, j]]).unsqueeze(0).float()
                    # Expand state and skill to match number of test actions
                    s_batch = s_torch.expand(len(test_actions), -1)
                    sk_batch = skill_vec.expand(len(test_actions), -1)
                    
                    with torch.no_grad():
                        qs = learner.q1(s_batch, test_actions, sk_batch)
                        max_q = qs.max().item()
                    all_q_data.append(max_q)
    
    q_min, q_max = np.nanmin(all_q_data), np.nanmax(all_q_data)
    print("Optimistic Q Range: [{0:.2f}, {1:.2f}]".format(q_min, q_max))

    # 5. Generate Gallery Plots
    fig, axes = plt.subplots(2, 5, figsize=(22, 6), sharex=True, sharey=True)
    axes = axes.flatten()
    
    print("--- Generating Optimistic Q-Gallery ---")
    for skill_idx in range(10):
        ax = axes[skill_idx]
        Q_map = np.full(X.shape, np.nan)
        skill_vec = learner.preprocess_skill(torch.tensor([skill_idx]))
        
        for i in range(res_y):
            for j in range(res_x):
                if not env.maze.is_inside_wall((X[i, j], Y[i, j])):
                    s_torch = torch.tensor([X[i, j], Y[i, j]]).unsqueeze(0).float()
                    s_batch = s_torch.expand(len(test_actions), -1)
                    sk_batch = skill_vec.expand(len(test_actions), -1)
                    with torch.no_grad():
                        qs = learner.q1(s_batch, test_actions, sk_batch)
                        Q_map[i, j] = qs.max().item()

        im = ax.imshow(Q_map, extent=[lims['x'][0], lims['x'][1], lims['y'][0], lims['y'][1]], 
                        origin='lower', cmap='magma', aspect='equal', vmin=q_min, vmax=q_max)
        
        # Add Contour Lines (White)
        # Fill NaNs with the minimum Q value to keep contour range meaningful
        Q_filled = np.nan_to_num(Q_map)
        Q_filled[np.isnan(Q_map)] = q_min
        ax.contour(X, Y, Q_filled, levels=15, colors='white', alpha=0.4, linewidths=0.5)
        
        g_phys = provider.get_goal_for_skill(skill_idx)
        ax.plot(g_phys[0], g_phys[1], 'r*', markersize=10, markeredgecolor='white')
        ax.set_title("Skill {0}".format(skill_idx), fontsize=12)
        ax.axis('equal')

    # 6. Final Polish
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Max Predicted Q-Value')
    fig.suptitle("SAC-v2 OPTIMISTIC Q-Value Gallery: {0}".format(args.maze_type), fontsize=18)
    
    save_path = "logs/inspection/{0}_q_gallery_optimistic.png".format(args.maze_type)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print("Saved Optimistic Q-Gallery to: {0}".format(save_path))

if __name__ == "__main__":

    run_optimistic_q_gallery()
