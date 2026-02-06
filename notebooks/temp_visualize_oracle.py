# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import argparse
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

try:
    from agents.maze_agents.toy_maze.env.maze_env import Env
    from result_inspection.toy_maze import config_subplot
    from geometry_aware_skill_discovery.generate_laplacian_oracle import Step
except ImportError as e:
    print("Import Error: {}. Run from 'notebooks' directory.".format(e))
    sys.exit(1)

def visualize_oracle_samples():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze_type", type=str, default="square_a")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--max_samples", type=int, default=0, help="If > 0, visualize N specific trajectories with colors.")
    args = parser.parse_args()

    print("Visualizing oracle data for: {}".format(args.maze_type))
    data_path = "../data/oracle_transitions/{}/{}/transitions.pkl".format(args.exp_name, args.maze_type)
    
    if not os.path.exists(data_path):
        print("Data file not found at {}".format(data_path))
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    transitions = data['raw_transitions']
    metadata = data.get('metadata', {})
    print("Loaded {} transitions.".format(len(transitions)))
    
    # Create environment for plotting
    env = Env(n=1, maze_type=args.maze_type, use_antigoal=False)
    
    if args.max_samples > 0 and 'random_walk_length' in metadata:
        # --- Trajectory Mode ---
        print("Trajectory mode: Plotting {} random walk paths.".format(args.max_samples))
        rw_len = metadata['random_walk_length']
        num_total_episodes = len(transitions) // rw_len
        
        episodes = []
        for i in range(num_total_episodes):
            ep_trans = transitions[i*rw_len : (i+1)*rw_len]
            path = [t[0] for t in ep_trans]
            path.append(ep_trans[-1][2]) 
            episodes.append(np.array(path))
            
        sample_indices = random.sample(range(num_total_episodes), min(args.max_samples, num_total_episodes))
        
        fig, ax = plt.subplots(figsize=(10, 10))
        env.maze.plot(ax)
        config_subplot(ax, maze_type=args.maze_type)
        
        for i, idx in enumerate(sample_indices):
            path = episodes[idx]
            color = "C{}".format(i % 10)
            # 0=X, 1=Y (as corrected)
            ax.plot(path[:, 0], path[:, 1], '-', color=color, alpha=0.8, linewidth=1.5)
            ax.scatter(path[0, 0], path[0, 1], color=color, s=30, edgecolors='black', zorder=5) 
            
        ax.set_title("Random Walk Trajectories ({}) - {}".format(args.max_samples, args.maze_type))
        save_name = "oracle_trajectories_{}.png".format(args.maze_type)
        plt.savefig(save_name)
        print("Saved {}".format(save_name))
        
    else:
        # --- Original Default Mode ---
        # Plot 1: All transitions as scatter points (start positions)
        fig, ax = plt.subplots(figsize=(10, 10))
        env.maze.plot(ax)
        config_subplot(ax, maze_type=args.maze_type)
        
        starts = np.array([t[0] for t in transitions])
        # 0=X, 1=Y (as corrected)
        ax.scatter(starts[:, 0], starts[:, 1], s=2, c='blue', alpha=0.3)
        ax.set_title("Oracle Sample Distribution - {} (Start Positions)".format(args.maze_type))
        save_name_scatter = "oracle_samples_scatter_{}.png".format(args.maze_type)
        plt.savefig(save_name_scatter)
        print("Saved {}".format(save_name_scatter))
        
        # Plot 2: Transitions as lines (Vector field)
        fig, ax = plt.subplots(figsize=(12, 12))
        env.maze.plot(ax)
        config_subplot(ax, maze_type=args.maze_type)
        
        if len(transitions) > 5000:
            indices = np.random.choice(len(transitions), 5000, replace=False)
            plot_transitions = [transitions[i] for i in indices]
        else:
            plot_transitions = transitions
            
        lines = []
        for s, a, s_next in plot_transitions:
            # 0=X, 1=Y (as corrected)
            lines.append([(s[0], s[1]), (s_next[0], s_next[1])])
            
        lc = mc.LineCollection(lines, colors='blue', linewidths=0.5, alpha=0.2)
        ax.add_collection(lc)
        ax.set_title("Oracle Transitions - {} (Vector Field)".format(args.maze_type))
        save_name_lines = "oracle_transitions_lines_{}.png".format(args.maze_type)
        plt.savefig(save_name_lines)
        print("Saved {}".format(save_name_lines))

if __name__ == "__main__":
    visualize_oracle_samples()