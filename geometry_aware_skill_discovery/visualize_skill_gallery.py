# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import sys
import matplotlib.pyplot as plt
import torch
import yaml
import pickle
import numpy as np
from argparse import ArgumentParser

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["ROOT_DIR"] = ".."

from base.learners.sac_v2 import SACV2Learner
from agents.maze_agents.toy_maze.env.maze_env import Env
from result_inspection.toy_maze import ENV_LIMS

def visualize_gallery():
    parser = ArgumentParser(description="Generate a Skill Gallery by overlaying rollouts for all skills.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    args = parser.parse_args()
    
    try:
        path_parts = args.model_path.split(os.sep)
        maze_type = path_parts[path_parts.index('rl') + 1]
    except (ValueError, IndexError):
        print("Could not auto-detect maze_type from path."); sys.exit(1)
    
    print("Visualizing All Skills for Maze: {0}".format(maze_type))

    # 1. Load Components
    with open('config.yaml', 'r') as f: config = yaml.load(f)
    exp_name = config['experiment']['exp_name']
    centroids_path = os.path.join("logs/spectral_kmeans", maze_type, exp_name, "intent_centroids.pkl")
    with open(centroids_path, 'rb') as f: data = pickle.load(f)
    centroids_s = data['centroids_s']; centroids_psi = torch.from_numpy(data['centroids_psi']).float()

    # 2. Setup Env and Agent
    env = Env(n=50, maze_type=maze_type, done_on_success=True)
    agent = SACV2Learner(env=env, hidden_size=256, skill_dim=centroids_psi.shape[1], skill_n=len(centroids_s))
    agent.load_checkpoint(args.model_path)
    agent.skill_embedding.weight.data.copy_(centroids_psi)
    agent.eval()

    # 3. Plotting Setup
    plt.figure(figsize=(8, 8)); ax = plt.gca()
    lims = ENV_LIMS[maze_type]
    cmap = plt.get_cmap('tab10')

    # Draw maze
    if hasattr(env.maze, '_walls'):
        for wall in env.maze._walls: ax.plot(wall[0], wall[1], 'k-', linewidth=3, alpha=0.6)
    
    # 4. Perform and Plot Rollouts
    for skill_idx in range(10):
        color = cmap(skill_idx)
        
        # [CRITICAL FIX] Reseed the environment on every skill to get new random start
        env.seed(skill_idx * 100) # Use a different seed for each skill
        
        goal_phys = centroids_s[skill_idx]
        env.reset(goal=goal_phys)
        state = env.state
        skill_psi = centroids_psi[skill_idx]
        
        trajectory = [state.numpy()]
        
        for _ in range(50):
            action = agent.select_action(state, skill_psi, deterministic=True)
            env.step(torch.from_numpy(action))
            state = env.state
            trajectory.append(state.numpy())
            if env.is_done: break
        
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=2.0, marker='.', markersize=4, alpha=0.8, label="Skill {0}".format(skill_idx))
        ax.plot(goal_phys[0], goal_phys[1], '*', color=color, markersize=15, markeredgecolor='black')

    # 5. Final Touches
    ax.set_title("SPECTRA All-Skill Gallery: {0}".format(maze_type), fontsize=16)
    ax.set_xlim(lims['x'][0] - 0.5, lims['x'][1] + 0.5); ax.set_ylim(lims['y'][0] - 0.5, lims['y'][1] + 0.5)
    ax.set_aspect('equal'); ax.grid(True, linestyle='--', alpha=0.4);
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    save_path = "logs/inspection/{0}_all_skills_gallery.png".format(maze_type)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print("\nAll-Skill Gallery saved to: {0}".format(save_path))

if __name__ == '__main__':
    visualize_gallery()
