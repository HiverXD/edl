# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle
from argparse import ArgumentParser

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.maze_agents.toy_maze.env.maze_env import Env
from base.learners.sac_v2 import SACV2Learner
from result_inspection.toy_maze import ENV_LIMS

def run_q_vs_pi_inspection():
    parser = ArgumentParser(description="Compare Optimal Q-Action vs Policy Action.")
    parser.add_argument("--maze_type", type=str, default="square_corridor")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--skill_idx", type=int, default=0)
    args = parser.parse_args()

    # 1. Load Intent Centroids DIRECTLY
    centroids_path = os.path.join("logs/spectral_kmeans", args.maze_type, "curriculum", "intent_centroids.pkl")
    with open(centroids_path, 'rb') as f:
        data = pickle.load(f)
    centroids_s = data['centroids_s']
    centroids_psi = torch.from_numpy(data['centroids_psi']).float()

    # 2. Setup Agent
    with open('config.yaml', 'r') as f: config = yaml.load(f)
    env = Env(n=1, maze_type=args.maze_type)
    agent = SACV2Learner(env=env, hidden_size=256, skill_dim=centroids_psi.shape[1], skill_n=len(centroids_s))
    agent.load_checkpoint(args.model_path)
    agent.skill_embedding.weight.data.copy_(centroids_psi)
    agent.eval()

    # 3. Generate Grid and Actions
    lims = ENV_LIMS[args.maze_type]
    res_x, res_y = 30, 8
    X, Y = np.meshgrid(np.linspace(lims['x'][0], lims['x'][1], res_x), np.linspace(lims['y'][0], lims['y'][1], res_y))
    
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    candidates = torch.tensor([[np.cos(a), np.sin(a)] for a in angles]).float() * env.action_range
    
    U_q, V_q = np.zeros_like(X), np.zeros_like(Y)
    U_pi, V_pi = np.zeros_like(X), np.zeros_like(Y)
    cos_sims = []
    
    skill_vec = agent.preprocess_skill(torch.tensor([args.skill_idx]))

    for i in range(res_y):
        for j in range(res_x):
            if env.maze.is_inside_wall((X[i, j], Y[i, j])): continue
            s_torch = torch.tensor([X[i, j], Y[i, j]]).unsqueeze(0).float()
            
            # A. Optimal Q
            with torch.no_grad():
                qs = agent.q1(s_torch.expand(8, -1), candidates, skill_vec.expand(8, -1))
                best_a = candidates[torch.argmax(qs)].numpy()
                U_q[i, j], V_q[i, j] = best_a[0], best_a[1]
            
            # B. Actual Policy
            with torch.no_grad():
                pi_a, _, _, _ = agent.actor(s_torch, skill_vec, greedy=True)
                pi_a = pi_a.squeeze(0).numpy()
                norm = np.linalg.norm(pi_a) + 1e-8
                u_p, v_p = pi_a[0]/norm, pi_a[1]/norm
                U_pi[i, j], V_pi[i, j] = u_p, v_p
                
                align = np.dot(best_a / (np.linalg.norm(best_a)+1e-8), [u_p, v_p])
                cos_sims.append(align)

    avg_align = np.mean(cos_sims)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    def plot_field(ax, U, V, title):
        if hasattr(env.maze, '_walls'):
            for wall in env.maze._walls: (wx, wy) = wall; ax.plot(wx, wy, 'k-', alpha=0.5)
        ax.quiver(X, Y, U, V, color='blue', alpha=0.7, scale=25)
        g_phys = centroids_s[args.skill_idx]
        ax.plot(g_phys[0], g_phys[1], 'r*', markersize=15)
        ax.set_title(title); ax.set_xlim(lims['x']); ax.set_ylim(lims['y']); ax.set_aspect('equal')

    plot_field(ax1, U_q, V_q, "1. Wise Critic: Max Q Direction (Skill {0})".format(args.skill_idx))
    plot_field(ax2, U_pi, V_pi, "2. Actor Behaviour (Align: {0:.4f})".format(avg_align))
    
    plt.tight_layout(); plt.savefig("logs/inspection/{0}_q_vs_pi.png".format(args.maze_type))
    print("Fixed Q vs Pi Alignment saved.")

if __name__ == "__main__":
    run_q_vs_pi_inspection()
