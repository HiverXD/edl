# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_aware_skill_discovery.laplacian_metric import LaplacianMetricCalculator
from geometry_aware_skill_discovery.spectral_k_means import SpectralKMeansManager
from agents.maze_agents.toy_maze.env.maze_env import Env
from result_inspection.toy_maze import ENV_LIMS, config_subplot

def main():
    parser = ArgumentParser(description="Visualize and compare Spectral K-means clustering methods.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--res", type=float, default=0.1, help="Grid resolution for sampling")
    parser.add_argument("--bandwidth", type=float, default=0.2, help="Bandwidth for KDE weighting")
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters (skills)")
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    exp_cfg = config['experiment']
    maze_type = exp_cfg['maze_type']
    exp_name = exp_cfg['exp_name']
    
    print("--- Spectral K-means Comparison for {0} ---".format(maze_type))

    # 2. Setup Env and Calculator
    env = Env(n=1, maze_type=maze_type, use_antigoal=False)
    calc_exp_name = os.path.join(exp_name, "stage_2") if exp_name == "curriculum" else exp_name
    calc = LaplacianMetricCalculator(maze_type=maze_type, exp_name=calc_exp_name)

    # 3. Generate Uniform Grid Samples
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
    valid_s = grid_points[valid_mask]
    
    # 4. Transform to Commute Space
    valid_psi = calc.transform_space(valid_s, mode="commute")
    
    # 5. Initialize Manager and Run Algorithms
    manager = SpectralKMeansManager(calc, n_clusters=args.n_clusters)
    weights = manager.get_kde_weights(valid_psi, bandwidth=args.bandwidth)
    
    # Method 1: Plain Bisecting K-means
    labels_b, centers_psi_b = manager.run_bisecting_kmeans(valid_psi)
    centers_s_b = manager.get_centroids_in_state_space(valid_psi, valid_s, centers_psi_b)
    
    # Method 2: Weighted K-means
    labels_w, centers_psi_w = manager.run_weighted_kmeans(valid_psi, weights)
    centers_s_w = manager.get_centroids_in_state_space(valid_psi, valid_s, centers_psi_w)
    
    # Method 3: Weighted Bisecting K-means
    labels_wb, centers_psi_wb = manager.run_bisecting_kmeans(valid_psi, weights=weights)
    centers_s_wb = manager.get_centroids_in_state_space(valid_psi, valid_s, centers_psi_wb)

    # 6. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(25, 8))
    titles = ["1. Bisecting (Hierarchical)", "2. Weighted K-means (KDE)", "3. Weighted Bisecting (H+KDE)"]
    all_labels = [labels_b, labels_w, labels_wb]
    all_centers = [centers_s_b, centers_s_w, centers_s_wb]
    
    cmap = plt.get_cmap('tab20', args.n_clusters)
    
    last_sc = None
    for i in range(3):
        ax = axes[i]
        env.maze.plot(ax)
        config_subplot(ax, maze_type=maze_type)
        
        # Color the grid points by label
        last_sc = ax.scatter(valid_s[:, 0], valid_s[:, 1], c=all_labels[i], cmap=cmap, s=5, alpha=0.6)
        
        # Plot centroids as stars
        ax.scatter(all_centers[i][:, 0], all_centers[i][:, 1], c='red', marker='*', s=200, edgecolors='black', label='Intent Centroids')
        
        ax.set_title(titles[i], fontsize=15)
            
    # Manually adjust subplots to leave room for the colorbar on the right
    plt.subplots_adjust(left=0.05, right=0.92, wspace=0.2, top=0.9, bottom=0.1)
    
    # Create a dedicated axis for the colorbar [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(last_sc, cax=cbar_ax)
    cbar.set_label("Intent (Skill) ID", fontsize=12)
            
    # Save the comparison
    save_dir = os.path.join("logs/spectral_kmeans", maze_type, exp_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "clustering_comparison.png")
    plt.savefig(save_path)
    print("\n[Done] Visualization saved to {0}".format(save_path))
    
    # Save Intent Centroids (Method 3: Weighted Bisecting) for Reward Function
    centroids_data = {
        'maze_type': maze_type,
        'n_clusters': args.n_clusters,
        'centroids_psi': centers_psi_wb, # Embedding space goals
        'centroids_s': centers_s_wb,     # Physical space goals (for reference)
        'method': 'Weighted Bisecting'
    }
    centroids_path = os.path.join(save_dir, "intent_centroids.pkl")
    import pickle
    with open(centroids_path, 'wb') as f:
        pickle.dump(centroids_data, f)
    print("[Done] Intent centroids saved to {0}".format(centroids_path))
    
    plt.show()

if __name__ == "__main__":
    main()

