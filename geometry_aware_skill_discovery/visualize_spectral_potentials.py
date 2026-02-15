# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from argparse import ArgumentParser
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_aware_skill_discovery.reward import SPECTRAProvider
from agents.maze_agents.toy_maze.env.maze_env import Env
from result_inspection.toy_maze import ENV_LIMS, config_subplot

def main():
    parser = ArgumentParser(description="Visualize SPECTRA Potential Fields for each intent.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--res", type=float, default=0.1, help="Grid resolution for heatmap")
    parser.add_argument("--ncols", type=int, default=5, help="Number of columns in subplot grid")
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    exp_cfg = config['experiment']
    maze_type = exp_cfg['maze_type']
    exp_name = exp_cfg['exp_name']
    
    print("--- Visualizing SPECTRA Potentials for {0} ---".format(maze_type))

    # 2. Setup SPECTRA Provider
    # This automatically loads Laplacian model and K-means centroids
    provider = SPECTRAProvider(maze_type=maze_type, exp_name=exp_name)
    n_skills = provider.n_skills

    # 3. Setup Environment and Grid Points
    env = Env(n=1, maze_type=maze_type, use_antigoal=False)
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
    valid_s = torch.from_numpy(grid_points[valid_mask]).float()

    # 4. Compute Potentials for all skills
    all_potentials = []
    print("Computing potential fields...")
    for k in range(n_skills):
        skill_idx = torch.full((len(valid_s),), k, dtype=torch.long)
        # Phi(s, g) = -0.5 * CTD(s, g)^2
        phi = provider.compute_potential(valid_s, skill_idx).numpy()
        all_potentials.append(phi)
    
    all_potentials = np.array(all_potentials)
    
    # 5. Determine Dynamic Range (Robust to outliers)
    # We use the 1st percentile of all data as vmin to get good contrast
    vmin = np.percentile(all_potentials, 1)
    vmax = 0 # Potentials are always negative or zero at target
    print("Dynamic Range: vmin={0:.2f}, vmax={1:.2f}".format(vmin, vmax))

    # 6. Plotting (Grid layout like original EDL paper)
    nrows = math.ceil(n_skills / args.ncols)
    fig, axarr = plt.subplots(ncols=args.ncols, nrows=nrows, figsize=(4 * args.ncols, 4.5 * nrows))
    axes = axarr.flatten()
    
    cmap = 'magma' # Dark-to-light heatmap
    
    for k in range(n_skills):
        ax = axes[k]
        env.maze.plot(ax)
        config_subplot(ax, maze_type=maze_type)
        
        # Reshape potentials to original 2D grid
        phi_2d = np.full(X.shape, np.nan)
        phi_2d[valid_mask.reshape(X.shape)] = all_potentials[k]
        
        # 1. Plot continuous heatmap using pcolormesh
        mesh = ax.pcolormesh(X, Y, phi_2d, vmin=vmin, vmax=vmax, cmap=cmap, shading='auto', alpha=0.9)
        
        # 2. Add contour lines for better topological structure visualization
        # Legacy compatible way to fill NaNs for contour plotting
        phi_contour = phi_2d.copy()
        phi_contour[np.isnan(phi_contour)] = vmin - 1
        # Explicitly define levels within range to avoid warnings
        levels = np.linspace(vmin, vmax, 10)
        ax.contour(X, Y, phi_contour, levels=levels, colors='white', alpha=0.3, linewidths=0.5)
        
        # 3. Plot the target centroid
        target_s = provider.centroids_s[k]
        ax.scatter(target_s[0], target_s[1], c='white', marker='*', s=150, edgecolors='black', linewidths=1.5, zorder=20, label='Target')
        
        ax.set_title("Potential $\Phi(s|g_{%d})$" % k, fontsize=14)

    # Turn off unused axes
    for idx in range(k + 1, len(axes)):
        axes[idx].axis('off')

    # Add a global colorbar
    plt.subplots_adjust(bottom=0.12, hspace=0.3, wspace=0.2)
    # Get the last mesh object for colorbar
    cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02])
    cbar = fig.colorbar(mesh, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Topological Potential ($\Phi$)", fontsize=12)

    # 7. Save and Finalize
    save_dir = os.path.join("logs/spectral_kmeans", maze_type, exp_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "spectra_potentials.png")
    plt.savefig(save_path)
    print("\n[Done] Potential visualization saved to {0}".format(save_path))
    plt.show()

if __name__ == "__main__":
    main()
