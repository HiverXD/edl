# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import yaml
from argparse import ArgumentParser

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_aware_skill_discovery.reward import SPECTRAProvider
from agents.maze_agents.toy_maze.env.maze_env import Env
from result_inspection.toy_maze import ENV_LIMS

def is_valid_point(maze, x, y):
    """Checks if a point is not inside a wall."""
    if hasattr(maze, 'is_inside_wall'):
        return not maze.is_inside_wall((x, y))
    # For mazes that don't have is_inside_wall (like CircleMaze), assume valid for now
    return True

def run_deep_inspection():
    parser = ArgumentParser(description="SPECTRA Deep Inspector: Analyzing Navigation Quality.")
    parser.add_argument("--maze_type", type=str, default="square_corridor")
    parser.add_argument("--exp_name", type=str, default="curriculum")
    parser.add_argument("--skill_idx", type=int, default=0)
    args = parser.parse_args()

    # 1. Setup Provider
    provider = SPECTRAProvider(maze_type=args.maze_type, exp_name=args.exp_name)
    env = Env(n=1, maze_type=args.maze_type, use_antigoal=False)
    
    g_phys = provider.get_goal_for_skill(args.skill_idx)
    if torch.is_tensor(g_phys): g_phys = g_phys.numpy()
    
    # 2. Resolve Environment Limits
    if args.maze_type in ENV_LIMS:
        lims = ENV_LIMS[args.maze_type]
        min_x, max_x = lims['x']
        min_y, max_y = lims['y']
    else:
        print("Warning: {0} not in ENV_LIMS. Using default [-5, 5].".format(args.maze_type))
        min_x, max_x = -5, 5
        min_y, max_y = -5, 5

    # 3. Dense Grid for Potential Heatmap
    viz_res = 50
    vx = np.linspace(min_x, max_x, viz_res)
    vy = np.linspace(min_y, max_y, viz_res)
    VX, VY = np.meshgrid(vx, vy)
    Z = np.full(VX.shape, np.nan)
    
    # 4. Sparse Grid for Quiver (Arrows)
    q_res = 20
    qx = np.linspace(min_x, max_x, q_res)
    qy = np.linspace(min_y, max_y, q_res)
    QX, QY = np.meshgrid(qx, qy)
    U = np.zeros_like(QX)
    V = np.zeros_like(QY)
    cos_sims = []

    print("\n--- A-1: Potential Field & Gradient Analysis ---")
    
    # Calculate Heatmap (Z)
    for i in range(viz_res):
        for j in range(viz_res):
            if not is_valid_point(env.maze, VX[i, j], VY[i, j]): continue
            s_torch = torch.tensor([VX[i, j], VY[i, j]]).unsqueeze(0).float()
            phi = provider.compute_potential(s_torch, torch.tensor([args.skill_idx]))
            Z[i, j] = phi.item()

    # Calculate Quiver (U, V)
    for i in range(q_res):
        for j in range(q_res):
            if not is_valid_point(env.maze, QX[i, j], QY[i, j]): continue
            s_torch = torch.tensor([QX[i, j], QY[i, j]]).unsqueeze(0).float()
            s_torch.requires_grad = True
            phi = provider.compute_potential(s_torch, torch.tensor([args.skill_idx]))
            phi.backward()
            if s_torch.grad is None: continue
            
            grad = s_torch.grad.squeeze(0).numpy()
            grad_unit = grad / (np.linalg.norm(grad) + 1e-8)
            
            ideal_dir = g_phys - np.array([QX[i, j], QY[i, j]])
            ideal_unit = ideal_dir / (np.linalg.norm(ideal_dir) + 1e-8)
            
            alignment = np.dot(ideal_unit, grad_unit)
            cos_sims.append(alignment)
            U[i, j], V[i, j] = grad_unit[0], grad_unit[1]

    if len(cos_sims) > 0:
        print("Average Gradient-Goal Alignment (Cosine Sim): {0:.4f}".format(np.mean(cos_sims)))

    # 5. Final Visualization
    plt.figure(figsize=(12, 10))
    
    # Potential Heatmap (Magma)
    # Using 'magma' as requested
    im = plt.imshow(Z, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap='magma', alpha=0.9)
    plt.colorbar(im, label='Potential Value (Phi)')
    
    # Contour Lines
    plt.contour(VX, VY, Z, levels=20, colors='white', alpha=0.2, linewidths=0.5)
    
    # Quiver Arrows (Cyan for visibility against magma)
    plt.quiver(QX, QY, U, V, color='cyan', alpha=0.8, scale=30, headwidth=3, label='Potential Gradient')
    
    # Goal (Large Red Star)
    plt.plot(g_phys[0], g_phys[1], 'r*', markersize=20, markeredgecolor='white', label='Goal')
    
    plt.title("SPECTRA Deep Inspection: {0} (Skill {1}) (Cos sim {2:.3f})".format(args.maze_type, args.skill_idx, np.mean(cos_sims)))
    plt.xlabel("X"); plt.ylabel("Y")
    plt.legend(loc='upper right')
    
    save_dir = "logs/inspection"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "{0}_skill{1}_deep.png".format(args.maze_type, args.skill_idx))
    plt.savefig(save_path, dpi=150)
    print("\nSaved Deep Diagnostic Plot to: {0}".format(save_path))

if __name__ == "__main__":
    run_deep_inspection()