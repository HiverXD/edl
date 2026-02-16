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

def run_q_vs_pi_inspection():
    parser = ArgumentParser(description="Compare Optimal Q-Action vs Policy Action.")
    parser.add_argument("--maze_type", type=str, default="square_corridor")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--skill_idx", type=int, default=0)
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

    # 3. Generate Grid and Actions
    lims = ENV_LIMS[args.maze_type]
    res_x, res_y = 30, 8
    x = np.linspace(lims['x'][0], lims['x'][1], res_x)
    y = np.linspace(lims['y'][0], lims['y'][1], res_y)
    X, Y = np.meshgrid(x, y)
    
    # 8-Direction candidates
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    candidates = torch.tensor([[np.cos(a), np.sin(a)] for a in angles]).float()
    
    # Data storage
    U_q, V_q = np.zeros_like(X), np.zeros_like(Y)
    U_pi, V_pi = np.zeros_like(X), np.zeros_like(Y)
    cos_sims = []
    
    skill_vec = learner.preprocess_skill(torch.tensor([args.skill_idx]))
    print("\n--- Comparing Q-Optimizer vs Policy (Skill {0}) ---".format(args.skill_idx))

    for i in range(res_y):
        for j in range(res_x):
            pos = (X[i, j], Y[i, j])
            if env.maze.is_inside_wall(pos):
                continue
                
            s_torch = torch.tensor([X[i, j], Y[i, j]]).unsqueeze(0).float()
            
            # A. Optimal Q Direction (Max Q over 8 directions)
            s_batch = s_torch.expand(len(candidates), -1)
            sk_batch = skill_vec.expand(len(candidates), -1)
            with torch.no_grad():
                qs = learner.q1(s_batch, candidates, sk_batch)
                best_idx = torch.argmax(qs).item()
                best_a = candidates[best_idx].numpy()
                U_q[i, j], V_q[i, j] = best_a[0], best_a[1]
            
            # B. Actual Policy Direction
            with torch.no_grad():
                pi_a, _, _, _ = learner.policy(s_torch, skill_vec)
                pi_a = pi_a.squeeze(0).numpy()
                norm = np.linalg.norm(pi_a) + 1e-8
                u_p, v_p = pi_a[0]/norm, pi_a[1]/norm
                U_pi[i, j], V_pi[i, j] = u_p, v_p
                
                # C. Cosine Similarity between Q-best and Policy-actual
                align = np.dot([best_a[0], best_a[1]], [u_p, v_p])
                cos_sims.append(align)

    avg_align = np.mean(cos_sims) if cos_sims else 0.0
    print("Average Q-Pi Alignment: {0:.4f}".format(avg_align))

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    def plot_field(ax, U, V, title):
        if hasattr(env.maze, '_walls'):
            for wall in env.maze._walls:
                (wx, wy) = wall
                ax.plot(wx, wy, 'k-', linewidth=2, alpha=0.5)
        ax.quiver(X, Y, U, V, color='blue', alpha=0.7, scale=25)
        g_phys = provider.get_goal_for_skill(args.skill_idx)
        ax.plot(g_phys[0], g_phys[1], 'r*', markersize=15, markeredgecolor='white')
        ax.set_title(title, fontsize=14)
        ax.set_xlim(lims['x']); ax.set_ylim(lims['y'])
        ax.set_aspect('equal')

    plot_field(ax1, U_q, V_q, "1. Wise Critic: Max Q Direction (Target Guidance)")
    plot_field(ax2, U_pi, V_pi, "2. Lost Actor: Policy Actual Direction (Align: {0:.4f})".format(avg_align))
    
    plt.tight_layout()
    save_path = "logs/inspection/{0}_q_vs_pi_skill{1}.png".format(args.maze_type, args.skill_idx)
    if not os.path.exists("logs/inspection"): os.makedirs("logs/inspection")
    plt.savefig(save_path, dpi=150)
    print("Comparison saved to: {0}".format(save_path))

if __name__ == "__main__":
    run_q_vs_pi_inspection()