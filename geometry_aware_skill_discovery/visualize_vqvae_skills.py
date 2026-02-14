# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_aware_skill_discovery.laplacian_metric import LaplacianMetricCalculator
from agents.maze_agents.toy_maze.env.maze_env import Env
from agents.maze_agents.toy_maze.skill_discovery.edl import VQVAEDiscriminator
from result_inspection.toy_maze import ENV_LIMS, config_subplot

def main():
    parser = ArgumentParser(description="Visualize VQ-VAE Skill Discovery in Maze Space.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--res", type=float, default=0.1, help="Grid resolution for visualization")
    args = parser.parse_args()

    # 1. Load Config and Resolve Paths
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    exp_cfg = config['experiment']
    vq_cfg = config['vqvae']
    maze_type = exp_cfg['maze_type']
    exp_name = exp_cfg['exp_name']
    mode = vq_cfg['mode']

    log_root = vq_cfg.get('log_dir', "logs/vqvae")
    exp_dir = os.path.join(log_root, maze_type, exp_name, mode)
    model_path = os.path.join(exp_dir, "model.pth.tar")

    if not os.path.exists(model_path):
        print("[Error] VQ-VAE model not found at {0}".format(model_path))
        return

    print("--- Visualizing VQ-VAE Skills (Mode: {0}) for {1} ---".format(mode, maze_type))

    # 2. Setup Environment and Laplacian Calculator
    env = Env(n=1, maze_type=maze_type, use_antigoal=False)
    
    if mode != "identity":
        calc_exp_name = os.path.join(exp_name, "stage_2") if exp_name == "curriculum" else exp_name
        calc = LaplacianMetricCalculator(maze_type=maze_type, exp_name=calc_exp_name)
    else:
        calc = None

    # 3. Load VQ-VAE Model
    # Determine state_size (2 for identity, dim for laplacian)
    state_size = 2 if mode == "identity" else exp_cfg['dim']
    
    # Load original training config to get vae_args
    with open(os.path.join(exp_dir, "config.json"), 'r') as f:
        train_config = json.load(f)
    
    model = VQVAEDiscriminator(state_size=state_size, **train_config['vae_args'])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 4. Generate Grid Points for Visualization
    try:
        lims = ENV_LIMS[maze_type]
        x_coords = np.arange(lims['x'][0], lims['x'][1], args.res)
        y_coords = np.arange(lims['y'][0], lims['y'][1], args.res)
    except KeyError:
        x_coords = np.arange(-5.5, 5.5, args.res)
        y_coords = np.arange(-5.5, 5.5, args.res)
    
    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    valid_mask = np.array([not env.maze.is_inside_wall(p) for p in grid_points])
    valid_points = grid_points[valid_mask]

    # 5. Transform and Pass through VQ-VAE
    if mode == "identity":
        transformed_points = valid_points
    else:
        transformed_points = calc.transform_space(valid_points, mode=mode)

    with torch.no_grad():
        batch = {"next_state": torch.from_numpy(transformed_points).float()}
        # compute_logprob returns (logprob, z_e, codes) when with_codes=True
        logprob, z_e, codes_raw = model.compute_logprob(batch, with_codes=True)
        
        # Convert codes to indices if they are one-hot or embedding vectors
        if len(codes_raw.shape) > 1:
            codes = codes_raw.argmax(dim=1).numpy()
        else:
            codes = codes_raw.numpy()
            
        # --- Diagnostic Prints ---
        unique_codes, counts = np.unique(codes, return_counts=True)
        print("\n[Diagnostic] Active Skills: {0} / {1}".format(len(unique_codes), train_config['vae_args']['codebook_size']))
        print("[Diagnostic] Skill Distribution: {0}".format(dict(zip(unique_codes.tolist(), counts.tolist()))))
        
        # Reconstruct to check error
        z_q, _ = model.vq.straight_through(z_e)
        reconstructed = model.decoder(z_q).numpy()
        
        recon_error = np.linalg.norm(transformed_points - reconstructed, axis=1)
        print("[Diagnostic] Recon Error (Phi-space): Mean={0:.4f}, Max={1:.4f}".format(np.mean(recon_error), np.max(recon_error)))
        # ------------------------
        
        # Calculate centroids by finding nearest neighbor in valid_points
        codebook = model.vq.embedding.weight.data.numpy() # (K, D_latent)
        # Centroids in state space
        centroids_indices = []
        centroids_coords = []
        
        # Decoder output for all codebook entries
        codebook_decoded = model.decoder(torch.from_numpy(codebook).float()).numpy() # (K, D_input)
        
        for k in range(codebook_decoded.shape[0]):
            target = codebook_decoded[k]
            # Find closest transformed point
            dists = np.linalg.norm(transformed_points - target, axis=1)
            nn_idx = np.argmin(dists)
            centroids_coords.append(valid_points[nn_idx])

    # 6. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Skill Assignments
    env.maze.plot(ax1)
    config_subplot(ax1, maze_type=maze_type)
    
    # Use discrete colormap for skills
    cmap = plt.get_cmap('tab20', train_config['vae_args']['codebook_size'])
    sc = ax1.scatter(valid_points[:, 0], valid_points[:, 1], c=codes, cmap=cmap, s=10, alpha=0.3)
    
    # Plot Centroids
    centroids_coords = np.array(centroids_coords)
    ax1.scatter(centroids_coords[:, 0], centroids_coords[:, 1], c='red', marker='*', s=150, edgecolors='black', label='Centroids')
    
    ax1.set_title("Skill Partitioning (Mode: {0})".format(mode))
    plt.colorbar(sc, ax=ax1, label="Skill ID")

    # Plot 2: Reconstruction Error
    env.maze.plot(ax2)
    config_subplot(ax2, maze_type=maze_type)
    
    recon_error = np.linalg.norm(transformed_points - reconstructed, axis=1)
    # Fill full grid for pcolormesh
    error_grid = np.full(X.size, np.nan)
    error_grid[valid_mask] = recon_error
    
    mesh = ax2.pcolormesh(X, Y, error_grid.reshape(X.shape), cmap='magma', shading='auto')
    ax2.set_title("Reconstruction Error (MSE in {0} space)".format(mode))
    plt.colorbar(mesh, ax=ax2, label="Error Magnitude")
    plt.tight_layout()

    save_path = os.path.join(exp_dir, "skill_visualization.png")
    plt.savefig(save_path)
    print("[Done] Visualization saved to {0}".format(save_path))
    plt.show()

if __name__ == "__main__":
    main()
