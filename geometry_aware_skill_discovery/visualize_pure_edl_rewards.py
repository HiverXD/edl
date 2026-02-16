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
from argparse import ArgumentParser

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.maze_agents.toy_maze.env.maze_env import Env
from agents.maze_agents.toy_maze.skill_discovery.edl import VQVAEDiscriminator
from result_inspection.toy_maze import ENV_LIMS

def run_pure_edl_reward_viz():
    parser = ArgumentParser(description="Visualize Pure EDL VQ-VAE Reward Map.")
    parser.add_argument("--maze_type", type=str, default="square_corridor")
    parser.add_argument("--vae_logdir", type=str, default="logs/vqvae/square_corridor/curriculum/identity")
    args = parser.parse_args()

    vae_path = os.path.join(args.vae_logdir, "model.pth.tar")
    with open(os.path.join(args.vae_logdir, "config.json"), 'r') as f: v_cfg = json.load(f)
    
    vae = VQVAEDiscriminator(state_size=2, **v_cfg['vae_args'])
    vae.load_state_dict(torch.load(vae_path, map_location='cpu'))
    vae.eval()
    
    env = Env(n=1, maze_type=args.maze_type)
    
    lims = ENV_LIMS[args.maze_type]
    res_x, res_y = 50, 15
    x = np.linspace(lims['x'][0], lims['x'][1], res_x)
    y = np.linspace(lims['y'][0], lims['y'][1], res_y)
    X, Y = np.meshgrid(x, y)
    
    fig, axes = plt.subplots(2, 5, figsize=(22, 6), sharex=True, sharey=True)
    axes = axes.flatten()
    
    print("\n--- Visualizing Pure EDL Rewards (-MSE) ---")
    
    for skill_idx in range(10):
        ax = axes[skill_idx]
        R_map = np.full(X.shape, np.nan)
        z_q = vae.vq.embedding(torch.tensor([skill_idx])).detach()
        x_target = vae.decoder(z_q).detach().numpy()
        
        for i in range(res_y):
            for j in range(res_x):
                if not env.maze.is_inside_wall((X[i, j], Y[i, j])):
                    pos = np.array([X[i, j], Y[i, j]])
                    mse = np.sum((pos - x_target)**2)
                    R_map[i, j] = -mse
        
        im = ax.imshow(R_map, extent=[lims['x'][0], lims['x'][1], lims['y'][0], lims['y'][1]], 
                        origin='lower', cmap='magma', aspect='equal')
        
        ax.plot(x_target[0, 0], x_target[0, 1], 'r*', markersize=10, markeredgecolor='white')
        ax.set_title("Skill {0}".format(skill_idx))
        ax.axis('equal')

    fig.suptitle("Original EDL Reward Geometries (-MSE)", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = "logs/inspection/{0}_pure_edl_rewards.png".format(args.maze_type)
    if not os.path.exists("logs/inspection"): os.makedirs("logs/inspection")
    plt.savefig(save_path, dpi=150)
    print("Saved Pure EDL Reward Map to: {0}".format(save_path))

if __name__ == "__main__":
    run_pure_edl_reward_viz()