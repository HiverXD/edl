# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.maze_agents.toy_maze.skill_discovery.edl import VQVAEDiscriminator
from agents.maze_agents.toy_maze.env.maze_env import Env
from result_inspection.toy_maze import ENV_LIMS, config_subplot

def main():
    parser = ArgumentParser(description="Visualize Ultra-Advanced VQ-VAE Skill Segmentation.")
    parser.add_argument("--maze_type", type=str, default="square_corridor")
    parser.add_argument("--seed", type=int, default=123, help="Seed to visualize")
    parser.add_argument("--res", type=float, default=0.1, help="Grid resolution for sampling")
    args = parser.parse_args()

    # 1. Setup paths
    exp_name = "curriculum"
    model_path = "logs/edl/advanced_vqvae/{0}/{1}/{2}/model.pth.tar".format(args.maze_type, exp_name, args.seed)
    save_dir = "logs/inspection"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "vqvae_ultra_segmentation_seed{0}.png".format(args.seed))

    # 2. Load Model
    model = VQVAEDiscriminator(state_size=2, hidden_size=256, codebook_size=10, code_size=2, normalize_inputs=True, decay=0.99)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 3. Grid Samples
    env = Env(n=1, maze_type=args.maze_type, use_antigoal=False)
    try:
        lims = ENV_LIMS[args.maze_type]
        x_coords = np.arange(lims['x'][0], lims['x'][1], args.res)
        y_coords = np.arange(lims['y'][0], lims['y'][1], args.res)
    except KeyError:
        x_coords = np.arange(-5.5, 5.5, args.res)
        y_coords = np.arange(-5.5, 5.5, args.res)
    
    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
    valid_mask = np.array([not env.maze.is_inside_wall(p) for p in grid_points])
    valid_s = grid_points[valid_mask]
    valid_s_torch = torch.from_numpy(valid_s).float()

    # 4. Predict
    with torch.no_grad():
        z_e_x = model.encoder(valid_s_torch)
        distances = model.vq.compute_distances(z_e_x)
        selected_indices = torch.argmin(distances, dim=1).numpy()
        w = model.vq.embedding.weight
        centroids_s = model.decoder(w)
        if model.normalizes_inputs:
            centroids_s = model.normalizer.denormalize(centroids_s)
        centroids_s = centroids_s.numpy()

    # 5. Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    env.maze.plot(ax)
    config_subplot(ax, maze_type=args.maze_type)
    cmap = plt.get_cmap('tab20', 10)
    sc = ax.scatter(valid_s[:, 0], valid_s[:, 1], c=selected_indices, cmap=cmap, s=15, alpha=0.6)
    ax.scatter(centroids_s[:, 0], centroids_s[:, 1], c='red', marker='*', s=300, edgecolors='black', label='VQ Centers')
    ax.set_title("Ultra-Advanced VQ-VAE (Seed {0})".format(args.seed), fontsize=18)
    
    plt.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Skill ID", fontsize=14)
    ax.legend(loc='upper right')
    
    plt.savefig(save_path)
    print("\n[Success] Saved to {0}".format(save_path))

if __name__ == "__main__":
    main()
