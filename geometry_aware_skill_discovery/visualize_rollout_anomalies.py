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
from argparse import ArgumentParser

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_aware_skill_discovery.reward import SPECTRAProvider
from agents.maze_agents.toy_maze.env.maze_env import Env
from agents import agent_classes

def analyze_rollout_anomalies():
    parser = ArgumentParser(description="Visualize Reward-Distance Anomalies & Policy Alignment.")
    parser.add_argument("--maze_type", type=str, default="square_corridor")
    parser.add_argument("--exp_name", type=str, default="curriculum")
    parser.add_argument("--reward_type", type=str, default="static")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--skill_idx", type=int, default=0)
    args = parser.parse_args()

    # 1. Setup Provider and Env
    provider = SPECTRAProvider(maze_type=args.maze_type, exp_name=args.exp_name)
    env = Env(n=50, maze_type=args.maze_type, done_on_success=True)
    
    # 2. Load Policy (GASD SAC-v2)
    AgentClass = agent_classes('maze', 'GASD', 'SAC_V2')
    with open('config.yaml', 'r') as f: config = yaml.load(f)
    agent_params = config['rl']['common'].copy()
    agent_params.update(config['rl'][args.reward_type])
    agent_params.update({
        'maze_type': args.maze_type, 'exp_name': args.exp_name, 
        'reward_type': args.reward_type, 'logging_keys': config['rl']['common'].get('logging_keys', [])
    })
    
    learner = AgentClass(**agent_params)
    print("Loading model from: {0}".format(args.model_path))
    learner.load_checkpoint(args.model_path)
    learner.eval()
    
    # 3. Perform Rollout with Anomaly & Alignment Detection
    goal_phys = provider.get_goal_for_skill(args.skill_idx)
    env.reset(goal=goal_phys)
    goal = env.goal
    
    history = []
    done = False
    step = 0
    alignments = []
    
    print("\n--- Starting Trace Rollout (Skill {0}) ---".format(args.skill_idx))
    
    while not done and step < 50:
        s_curr_np = env.state.numpy()
        d_curr = env.dist(env.state, goal).item()
        
        # Get Action and compute Gradient
        s_curr_torch = env.state.unsqueeze(0).float()
        s_curr_torch.requires_grad = True
        
        # Action from Policy
        with torch.no_grad():
            action, _, _, _ = learner.policy(
                s_curr_torch,
                learner.preprocess_skill(torch.tensor([args.skill_idx]))
            )
        action_np = action.squeeze(0).detach().numpy()
        
        # Potential Gradient: dPhi / dS
        phi_val = provider.compute_potential(s_curr_torch, torch.tensor([args.skill_idx]))
        phi_val.backward()
        grad = s_curr_torch.grad.squeeze(0).numpy()
        
        # Alignment Calculation (Action vs Gradient)
        a_unit = action_np / (np.linalg.norm(action_np) + 1e-8)
        g_unit = grad / (np.linalg.norm(grad) + 1e-8)
        policy_map_align = np.dot(a_unit, g_unit)
        alignments.append(policy_map_align)
        
        # Physical Step
        env.step(torch.from_numpy(action_np))
        s_next = env.state
        d_next = env.dist(s_next, goal).item()
        
        phi_next = provider.compute_potential(s_next.unsqueeze(0).float(), torch.tensor([args.skill_idx])).item()
        dist_change = d_next - d_curr
        
        status = "normal"
        if len(history) > 0:
            if dist_change < -0.01 and phi_next < history[-1]['phi']:
                status = "anomaly"
        if abs(dist_change) < 0.001 and np.abs(action_np).sum() > 0.1:
            status = "wall_hit"
            
        history.append({
            'step': step, 'pos': s_curr_np, 'dist': d_curr, 'phi': phi_next, 
            'action': action_np, 'grad': grad, 'status': status, 'align': policy_map_align
        })
        done = env.is_done; step += 1

    print("Average Policy-Map Alignment: {0:.4f}".format(np.mean(alignments)))

    # 4. Dual-Axis Plot (Unchanged + Alignment shading)
    steps = [h['step'] for h in history]
    dists = [h['dist'] for h in history]
    phis = [h['phi'] for h in history]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax1.plot(steps, dists, 'b-o', label='Physical Distance')
    ax2.plot(steps, phis, 'r-s', label='SPECTRA Potential')
    ax1.set_xlabel('Steps'); ax1.set_ylabel('Distance', color='b'); ax2.set_ylabel('Potential (Phi)', color='r')
    plt.title("Reward-Distance Correlation Trace (Skill {0})".format(args.skill_idx))
    plt.grid(True, alpha=0.3)
    
    for h in history:
        if h['status'] == "anomaly":
            ax1.axvspan(h['step']-0.5, h['step']+0.5, color='red', alpha=0.2)
            
    save_dir = "logs/inspection"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, "rollout_correlation.png"))
    
    # 5. Trajectory Map (Unchanged + Action Arrows)
    from result_inspection.toy_maze import ENV_LIMS
    plt.figure(figsize=(12, 3))
    lims = ENV_LIMS[args.maze_type]
    
    if hasattr(env.maze, '_walls'):
        for wall in env.maze._walls:
            (wx, wy) = wall
            plt.plot(wx, wy, 'k-', linewidth=3, alpha=0.6)
    
    for i in range(len(history)-1):
        p1, p2 = history[i]['pos'], history[i+1]['pos']
        color = 'green' if history[i]['status'] == 'normal' else ('red' if history[i]['status'] == 'anomaly' else 'blue')
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2, marker='.', markersize=4)
        
        # Add small arrows for Policy Actions
        if i % 2 == 0: # Every 2 steps to avoid clutter
            a = history[i]['action'] * 0.5 # Scale for visibility
            plt.arrow(p1[0], p1[1], a[0], a[1], head_width=0.05, color='orange', alpha=0.5)

    plt.plot(history[0]['pos'][0], history[0]['pos'][1], 'go', label='Start')
    goal_np = goal.numpy()
    plt.plot(goal_np[0], goal_np[1], 'r*', markersize=15, label='Goal')
    plt.title("Trajectory: Green=OK, Red=Anomaly, Blue=WallHit, Orange=Policy Action")
    plt.xlim(lims['x'][0]-0.5, lims['x'][1]+0.5); plt.ylim(lims['y'][0]-0.5, lims['y'][1]+0.5)
    plt.axis('equal'); plt.legend(loc='upper right', fontsize='small')
    plt.savefig(os.path.join(save_dir, "rollout_trajectory.png"), bbox_inches='tight')
    
    print("\nVisual Analysis Completed.")
    print("- Average Policy-Map Alignment: {0:.4f}".format(np.mean(alignments)))
    print("- Look for ORANGE arrows in rollout_trajectory.png to see agent's intent.")

if __name__ == "__main__":
    analyze_rollout_anomalies()
