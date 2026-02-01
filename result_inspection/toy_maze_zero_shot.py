# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import json
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm, tqdm_notebook

from .toy_maze import config_subplot, play_interpolated_episode

def rollout_arbitrary_zero_shot(exp, start_pos, goal_pos, base_save_dir="arbitrary_zero_shot_logs", num_rollouts=10, start_pos_random_range=0.2):
    """
    Performs rollouts from a start position towards an arbitrary goal state and saves trajectories.

    Args:
        exp (Experiment): The experiment object.
        start_pos (tuple): The center of the area for starting positions.
        goal_pos (tuple): The target goal position.
        base_save_dir (str): The root directory to save logs.
        num_rollouts (int): The number of trajectories to generate.
        start_pos_random_range (float): The range for randomizing start positions.
    """
    agent = exp.learner.agent
    vae = exp.learner.vae
    env = exp.learner.agent.env

    # 1. Validate start and goal positions
    if env.maze.is_inside_wall(start_pos):
        print("Error: The start position {} is inside a wall.".format(start_pos))
        return
    if env.maze.is_inside_wall(goal_pos):
        print("Error: The goal position {} is inside a wall.".format(goal_pos))
        return

    # 2. Create Log Directory
    start_pos_str = "{:.1f}_{:.1f}".format(start_pos[0], start_pos[1])
    goal_pos_str = "{:.1f}_{:.1f}".format(goal_pos[0], goal_pos[1])
    log_dir = os.path.join(base_save_dir, exp.name, "s_{}_g_{}".format(start_pos_str, goal_pos_str))
    
    shutil.rmtree(log_dir, ignore_errors=True)
    os.makedirs(log_dir, exist_ok=True)
    print("Saving arbitrary zero-shot trajectories to: {}".format(log_dir))

    # 3. Encode Goal State
    with torch.no_grad():
        goal_pos_tensor = torch.tensor(goal_pos, dtype=torch.float32).unsqueeze(0)
        z_goal = vae.encoder(goal_pos_tensor)

    # 4. Rollout Loop
    episodes_data = {}
    for k in range(num_rollouts):
        print("  Collecting trajectory {}/{}".format(k + 1, num_rollouts))

        # Randomize start position
        randomized_start_x = start_pos[0] + np.random.uniform(-start_pos_random_range, start_pos_random_range)
        randomized_start_y = start_pos[1] + np.random.uniform(-start_pos_random_range, start_pos_random_range)
        reset_dict = {'state': (randomized_start_x, randomized_start_y)}

        # Play episode with the goal-encoded skill
        play_interpolated_episode(agent, z_goal, do_eval=True, reset_dict=reset_dict)

        # Store trajectory data
        dump_ep = []
        for t_step in agent.episode:
            dump_t = {}
            for key, val in t_step.items():
                if isinstance(val, torch.Tensor):
                    dump_t[key] = val.detach().numpy().tolist()
                elif isinstance(val, np.ndarray):
                    dump_t[key] = val.tolist()
                else:
                    dump_t[key] = val
            dump_ep.append(dump_t)
        episodes_data[str(k)] = dump_ep

    # 5. Save all trajectories to a single JSON file
    traj_file_path = os.path.join(log_dir, "trajectories.json")
    with open(traj_file_path, 'wt') as f:
        json.dump(episodes_data, f, indent=2)
    
    print("Arbitrary zero-shot rollouts completed and saved.")
    return log_dir


def visualize_arbitrary_zero_shot(exp, log_dir, start_pos, goal_pos, ax=None, **kwargs):
    """
    Visualizes trajectories from `rollout_arbitrary_zero_shot` and calculates L2 error.

    Args:
        exp (Experiment): The experiment object.
        log_dir (str): The directory containing the trajectory logs.
        start_pos (tuple): The center of the starting area.
        goal_pos (tuple): The target goal position.
        ax (matplotlib.axes.Axes, optional): The axes to plot on.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 1. Setup plot
    config_subplot(ax, exp=exp, **kwargs)
    exp.learner.agent.env.maze.plot(ax)

    # 2. Load trajectories
    traj_file = os.path.join(log_dir, "trajectories.json")
    if not os.path.exists(traj_file):
        print("Error: trajectories.json not found in {}.".format(log_dir))
        return

    with open(traj_file, 'r') as f:
        episodes_data = json.load(f)

    # 3. Plot trajectories and calculate errors
    distances = []
    for k_str, trajectory_data in episodes_data.items():
        if not trajectory_data:
            continue
        
        # Plot trajectory
        states_x = [step['state'][0] for step in trajectory_data]
        states_y = [step['state'][1] for step in trajectory_data]
        ax.plot(states_x, states_y, color='c', alpha=0.5, linewidth=2)

        # Plot final achieved position
        s_final = np.array(trajectory_data[-1]['state'])
        ax.plot(s_final[0], s_final[1], 'o', color='darkorange', markersize=5, alpha=0.8)

        # Calculate L2 distance
        dist = np.linalg.norm(np.array(goal_pos) - s_final)
        distances.append(dist)

    # 4. Plot start and goal markers
    ax.plot(start_pos[0], start_pos[1], 'bo', markersize=12, label='Start Area Center', zorder=11, markeredgecolor='black')
    ax.plot(goal_pos[0], goal_pos[1], '*', color='red', markersize=18, label='Target Goal', zorder=12, markeredgecolor='black')

    # 5. Report statistics
    if distances:
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        stats_text = "Mean L2 Error: {:.3f}\nStd Dev: {:.3f}".format(mean_dist, std_dist)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend(loc='upper right')
    ax.set_title("Zero-Shot Rollouts from {} to {}".format(start_pos, goal_pos))
    
    return ax