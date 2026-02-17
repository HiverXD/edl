# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import json
import pickle
from argparse import ArgumentParser

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_aware_skill_discovery.reward import SPECTRAProvider
from agents.maze_agents.toy_maze.env.maze_env import Env
from base.learners.sac_v2 import SACV2Learner
from result_inspection.toy_maze import ENV_LIMS

def run_optimistic_q_gallery():
    parser = ArgumentParser(description="Visualize Optimistic Q-Value Gallery.")
    parser.add_argument("--maze_type", type=str, default="square_corridor")
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    # 1. Load Intent Centroids DIRECTLY for absolute consistency
    centroids_path = os.path.join("logs/spectral_kmeans", args.maze_type, "curriculum", "intent_centroids.pkl")
    with open(centroids_path, 'rb') as f:
        data = pickle.load(f)
    centroids_s = data['centroids_s']
    centroids_psi = torch.from_numpy(data['centroids_psi']).float()
    print("Directly loaded {0} centroids for visualization.".format(len(centroids_s)))

    # 2. Setup Provider and Env
    with open('config.yaml', 'r') as f: config = yaml.load(f)
    # Ensure provider uses the same centroids
    provider = SPECTRAProvider(maze_type=args.maze_type, exp_name="curriculum")
    env = Env(n=1, maze_type=args.maze_type)
    
    # 3. Load Agent
    common = config['rl']['common']
    agent = SACV2Learner(env=env, hidden_size=256, skill_dim=centroids_psi.shape[1], skill_n=len(centroids_s))
    agent.load_checkpoint(args.model_path)
    # Sync agent's embedding with the loaded centroids just in case
    agent.skill_embedding.weight.data.copy_(centroids_psi)
    agent.eval()

    # 4. Generate Grid
    lims = ENV_LIMS[args.maze_type]
    res_x, res_y = 40, 10
    X, Y = np.meshgrid(np.linspace(lims['x'][0], lims['x'][1], res_x), np.linspace(lims['y'][0], lims['y'][1], res_y))
    
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    test_actions = torch.tensor([[np.cos(a), np.sin(a)] for a in angles]).float() * env.action_range

    # 5. Calculate Global Q Range
    all_q_data = []
    for skill_idx in range(10):
        skill_vec = agent.preprocess_skill(torch.tensor([skill_idx]))
        for i in range(res_y):
            for j in range(res_x):
                if not env.maze.is_inside_wall((X[i, j], Y[i, j])):
                    s_torch = torch.tensor([X[i, j], Y[i, j]]).unsqueeze(0).float()
                    with torch.no_grad():
                        qs = agent.q1(s_torch.expand(8, -1), test_actions, skill_vec.expand(8, -1))
                        all_q_data.append(qs.max().item())
    
    q_min, q_max = np.nanmin(all_q_data), np.nanmax(all_q_data)

    # 6. Plot Gallery
    fig, axes = plt.subplots(2, 5, figsize=(22, 6), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for skill_idx in range(10):
        ax = axes[skill_idx]
        Q_map = np.full(X.shape, np.nan)
        skill_vec = agent.preprocess_skill(torch.tensor([skill_idx]))
        
        for i in range(res_y):
            for j in range(res_x):
                if not env.maze.is_inside_wall((X[i, j], Y[i, j])):
                    s_torch = torch.tensor([X[i, j], Y[i, j]]).unsqueeze(0).float()
                    with torch.no_grad():
                        qs = agent.q1(s_torch.expand(8, -1), test_actions, skill_vec.expand(8, -1))
                        Q_map[i, j] = qs.max().item()

        im = ax.imshow(Q_map, extent=[lims['x'][0], lims['x'][1], lims['y'][0], lims['y'][1]], 
                        origin='lower', cmap='magma', vmin=q_min, vmax=q_max)
        
        # Use DIRECTLY loaded coordinate
        g_phys = centroids_s[skill_idx]
        ax.plot(g_phys[0], g_phys[1], 'r*', markersize=12, markeredgecolor='white')
        ax.set_title("Skill {0}".format(skill_idx))
        ax.axis('equal')

    fig.subplots_adjust(right=0.9)
    fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.015, 0.7]))
    plt.savefig("logs/inspection/{0}.png".format(args.maze_type), bbox_inches='tight', dpi=150)
    print("Fixed Q-Gallery saved.")

if __name__ == "__main__":
    run_optimistic_q_gallery()
