# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import numpy as np
import matplotlib.pyplot as plt
import torch

# Define ENV_LIMS here to break circular dependency

ENV_LIMS = dict(
    square_a=dict(xlim=(-0.55, 4.55), ylim=(-4.55, 0.55), x=(-0.5, 4.5), y=(-4.5, 0.5)),
    square_b=dict(xlim=(-0.55, 4.55), ylim=(-4.55, 0.55), x=(-0.5, 4.5), y=(-4.5, 0.5)),
    square_c=dict(xlim=(-0.55, 4.55), ylim=(-4.55, 0.55), x=(-0.5, 4.5), y=(-4.5, 0.5)),
    square_bottleneck=dict(xlim=(-0.55, 9.55), ylim=(-0.55, 9.55), x=(-0.5, 9.5), y=(-0.5, 9.5)),
    square_corridor=dict(xlim=(-5.55, 5.55), ylim=(-0.55, 0.55), x=(-5.5, 5.5), y=(-0.5, 0.5)),
    square_corridor2=dict(xlim=(-5.55, 5.55), ylim=(-0.55, 0.55), x=(-5.5, 5.5), y=(-0.5, 0.5)),
    square_tree=dict(xlim=(-6.55, 6.55), ylim=(-6.55, 0.55), x=(-6.5, 6.5), y=(-6.5, 0.5)),
    spiral=dict(xlim=(-0.55, 5.55), ylim=(-4.55, 0.55), x=(-0.5, 5.5), y=(-4.5, 0.5)),
    large_spiral=dict(xlim=(-8.55, 0.55), ylim=(-0.55, 8.55), x=(-8.5, 0.5), y=(-0.5, 8.5)),
    square_ant_maze_1=dict(xlim=(-0.55, 9.55), ylim=(-0.55, 6.55), x=(-0.5, 9.5), y=(-0.5, 6.5)),
    square_large=dict(xlim=(-0.55, 9.55), ylim=(-0.55, 9.55), x=(-0.5, 9.5), y=(-0.5, 9.5)),
)

def config_subplot(ax, maze_type="square_a", exp=None):
    if exp:
        maze_type = exp.learner.agent.env.maze_type
    
    try:
        lims = ENV_LIMS[maze_type]
        ax.set_xlim(lims['xlim'])
        ax.set_ylim(lims['ylim'])
    except KeyError:
        pass
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

def visualize_laplacian_embedding(env, calc, grid_resolution=0.1, t=1.0, ax_arr=None):
    """
    Visualizes the Laplacian embedding of the maze state space in 4 ways.
    """
    if ax_arr is None:
        fig, ax_arr = plt.subplots(1, 4, figsize=(24, 6))
        
    ax1, ax2, ax3, ax4 = ax_arr
    
    # 1. Generate Valid Grid Points
    maze_type = env.maze_type
    try:
        env_lims = ENV_LIMS[maze_type]
        min_x, max_x = env_lims['x']
        min_y, max_y = env_lims['y']
    except KeyError:
        raise Exception('key error, add toy_maze.py, ENV_LIMS')
        
    x_coords = np.arange(min_x, max_x, grid_resolution)
    y_coords = np.arange(min_y, max_y, grid_resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    valid_points = []
    for p in grid_points:
        if not env.maze.is_inside_wall(p):
            valid_points.append(p)
    valid_points = np.array(valid_points)
    
    if len(valid_points) == 0:
        print("No valid points found.")
        return
        
    # 2. Assign Rainbow Colors
    norm_x = (valid_points[:, 0] - min_x) / (max_x - min_x)
    norm_y = (valid_points[:, 1] - min_y) / (max_y - min_y)
    color_index = (norm_x + norm_y) / 2
    cmap = plt.get_cmap('jet')
    colors = cmap(color_index)
    
    # 3. Transform Spaces using LaplacianMetricCalculator
    space_truncated = calc.transform_space(valid_points, mode="truncated")
    space_commute = calc.transform_space(valid_points, mode="commute")
    space_diffusion = calc.transform_space(valid_points, mode="diffusion", t=t)
        
    # 4. Plot 1: Maze Grid
    env.maze.plot(ax1)
    ax1.scatter(valid_points[:, 0], valid_points[:, 1], c=colors, s=2, alpha=0.8)
    ax1.set_title("Coordinate Space (Rainbow)", fontsize=12)
    ax1.axis('equal')
    
    # 5. Plot 2: Truncated Embedding (phi_1 vs phi_2)
    ax2.scatter(space_truncated[:, 0], space_truncated[:, 1], c=colors, s=2, alpha=0.8)
    ax2.set_title("Truncated ($\phi_1$ vs $\phi_2$)", fontsize=12)
    ax2.set_xlabel("$\phi_1$", fontsize=10)
    ax2.set_ylabel("$\phi_2$", fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 6. Plot 3 & 4: t-SNE
    try:
        from sklearn.manifold import TSNE
        print("Computing t-SNE for Commute and Diffusion spaces...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(50, len(valid_points)-1))
        
        phi_tsne_commute = tsne.fit_transform(space_commute)
        ax3.scatter(phi_tsne_commute[:, 0], phi_tsne_commute[:, 1], c=colors, s=2, alpha=0.8)
        ax3.set_title("t-SNE (Commute Space)", fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        phi_tsne_diffusion = tsne.fit_transform(space_diffusion)
        ax4.scatter(phi_tsne_diffusion[:, 0], phi_tsne_diffusion[:, 1], c=colors, s=2, alpha=0.8)
        ax4.set_title("t-SNE (Diffusion Space, t={0})".format(t), fontsize=12)
        ax4.grid(True, alpha=0.3)
        
    except ImportError:
        for ax in [ax3, ax4]: ax.text(0.5, 0.5, "sklearn not installed", ha='center')
        
    return ax_arr