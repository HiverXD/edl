# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import yaml
import pickle
from argparse import ArgumentParser

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_aware_skill_discovery.reward import SPECTRAProvider
from agents.maze_agents.toy_maze.env.maze_env import Env
from base.learners.sac_v2 import SACV2Learner

def analyze_rollout_anomalies():
    parser = ArgumentParser(description="Visualize Reward-Distance Anomalies & Rainbow Trajectory.")
    parser.add_argument("--maze_type", type=str, default="square_corridor")
    parser.add_argument("--exp_name", type=str, default="curriculum")
    parser.add_argument("--reward_type", type=str, default="static")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--skill_idx", type=int, default=0)
    args = parser.parse_args()

    # 1. Load Intent Centroids
    centroids_path = os.path.join("logs/spectral_kmeans", args.maze_type, args.exp_name, "intent_centroids.pkl")
    with open(centroids_path, 'rb') as f:
        data = pickle.load(f)
    centroids_s = data['centroids_s']
    centroids_psi = torch.from_numpy(data['centroids_psi']).float()

    # 2. Setup Provider and Env
    with open('config.yaml', 'r') as f: config = yaml.load(f)
    provider = SPECTRAProvider(maze_type=args.maze_type, exp_name=args.exp_name)
    env = Env(n=50, maze_type=args.maze_type, done_on_success=True)
    
    # 3. Load Agent
    common = config['rl']['common']
    agent = SACV2Learner(env=env, hidden_size=256, skill_dim=centroids_psi.shape[1], skill_n=len(centroids_s))
    agent.load_checkpoint(args.model_path)
    agent.skill_embedding.weight.data.copy_(centroids_psi)
    agent.eval()
    
    # 4. Perform Rollout
    goal_phys = centroids_s[args.skill_idx]
    env.reset(goal=goal_phys)
    goal = env.goal
    
    history = []
    done = False
    step = 0
    alignments = []
    
    while step < 50:
        s_curr = env.state
        d_curr = env.dist(s_curr, goal).item()
        s_curr_torch = s_curr.unsqueeze(0).float(); s_curr_torch.requires_grad = True
        
        with torch.no_grad():
            action, _, _, _ = agent.actor(s_curr_torch, agent.preprocess_skill(torch.tensor([args.skill_idx])), greedy=True)
        action = action.squeeze(0); action_np = action.detach().numpy()
        
        phi_val = provider.compute_potential(s_curr_torch, torch.tensor([args.skill_idx]))
        phi_val.backward()
        grad = s_curr_torch.grad.squeeze(0).numpy()
        
        a_unit = action_np / (np.linalg.norm(action_np) + 1e-8)
        g_unit = grad / (np.linalg.norm(grad) + 1e-8)
        policy_map_align = np.dot(a_unit, g_unit)
        alignments.append(policy_map_align)
        
        # Step the environment
        env.step(action)
        s_next = env.state
        d_next = env.dist(s_next, goal).item()
        phi_next = provider.compute_potential(s_next.unsqueeze(0).float(), torch.tensor([args.skill_idx])).item()
        
        history.append({'step': step, 'pos': s_curr.numpy(), 'dist': d_curr, 'phi': phi_next, 'action': action_np, 'align': policy_map_align})
        
        # [FIX] Check done AFTER recording current transition
        if env.is_done:
            # Also record the final next_state if it's a success
            if env.is_success:
                history.append({'step': step+1, 'pos': s_next.numpy(), 'dist': d_next, 'phi': phi_next, 'action': np.zeros(2), 'align': 1.0})
            break
        step += 1

    avg_align = np.mean(alignments)
    print("Average Policy-Map Alignment: {0:.4f}".format(avg_align))

    # 5. Dual-Axis Plot (Distance includes 0)
    steps = [h['step'] for h in history]
    dists = [h['dist'] for h in history]
    phis = [h['phi'] for h in history]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax1.plot(steps, dists, 'b-o', label='Physical Distance')
    ax2.plot(steps, phis, 'r-s', label='Potential (Phi)')
    ax1.set_xlabel('Steps'); ax1.set_ylabel('Distance', color='b'); ax2.set_ylabel('Potential', color='r')
    ax1.set_ylim(bottom=0) # [NEW] Force Distance to show 0
    plt.title("Correlation Trace (Align: {0:.4f})".format(avg_align))
    plt.grid(True, alpha=0.3)
    
    save_dir = "logs/inspection"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, "{0}_skill{1}_rollout_correlation.png".format(args.maze_type, args.skill_idx)))
    
    # 6. Trajectory Map (Reverted to Red for clarity)
    from result_inspection.toy_maze import ENV_LIMS
    plt.figure(figsize=(12, 3))
    lims = ENV_LIMS[args.maze_type]
    if hasattr(env.maze, '_walls'):
        for wall in env.maze._walls: (wx, wy) = wall; plt.plot(wx, wy, 'k-', linewidth=3, alpha=0.4)
    
    # Plot Trajectory in solid red with dots
    for i in range(len(history)-1):
        p1, p2 = history[i]['pos'], history[i+1]['pos']
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=2.0, marker='.', markersize=4, alpha=0.8)
        
    plt.plot(history[0]['pos'][0], history[0]['pos'][1], 'go', markersize=10, label='Start')
    plt.plot(goal.numpy()[0], goal.numpy()[1], 'r*', markersize=15, markeredgecolor='white', label='Goal')
    plt.title("{0} Skill {1} Rollout Trajectory".format(args.maze_type, args.skill_idx))
    plt.xlim(lims['x'][0]-0.5, lims['x'][1]+0.5); plt.ylim(lims['y'][0]-0.5, lims['y'][1]+0.5)
    plt.axis('equal'); plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_dir, "{0}_skill{1}_rollout_trajectory.png".format(args.maze_type, args.skill_idx)), bbox_inches='tight')
    
    print("Visual Analysis Completed. Standardized files saved.")

if __name__ == "__main__":
    analyze_rollout_anomalies()
