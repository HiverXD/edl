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


def rollout_interpolated_zero_shot(exp, skill_idx_1, skill_idx_2, num_interpolation, num_rollout_traj_each_mode, base_save_dir="zero_shot_logs", start_pos_base=(0.0, -0.5), random_range=0.2):
    """
    Performs rollouts for interpolated skills and saves trajectories in JSON format.
    """
    agent = exp.learner.agent
    vae = exp.learner.vae

    # Get base skill vectors
    z1 = agent.skill_embedding(torch.tensor(skill_idx_1))
    z2 = agent.skill_embedding(torch.tensor(skill_idx_2))

    # Q2 Fix: Clear previous logs for this experiment
    zero_shot_log_dir = os.path.join(base_save_dir, exp.name, "zero_shot_of_{}_{}".format(skill_idx_1, skill_idx_2))
    shutil.rmtree(zero_shot_log_dir, ignore_errors=True)
    os.makedirs(zero_shot_log_dir, exist_ok=True)
    
    print("Saving zero-shot trajectories to: {}".format(zero_shot_log_dir))

    # Iterate interpolation steps
    for i in range(num_interpolation + 2):
        # Calculate interpolation weight t
        t = float(i) / (num_interpolation + 1)
        
        # Calculate interpolated skill vector z_inter
        z_inter = (1 - t) * z1 + t * z2

        # Create subdirectory for this interpolated skill
        sub_dir_name = "{}-{}".format(i, num_interpolation + 1)
        skill_save_path = os.path.join(zero_shot_log_dir, sub_dir_name)
        os.makedirs(skill_save_path, exist_ok=True)

        episodes_data = {}
        # Collect multiple trajectories for each interpolated skill
        for k in range(num_rollout_traj_each_mode):
            print("  Collecting trajectory {}/{} for skill {}/{} (t={:.2f})".format(k+1, num_rollout_traj_each_mode, i, num_interpolation + 1, t))
            
            # Q1 Fix: Use a randomized starting state for each trajectory
            randomized_start_x = start_pos_base[0] + np.random.uniform(-random_range, random_range)
            randomized_start_y = start_pos_base[1] + np.random.uniform(-random_range, random_range)
            reset_dict = {'state': (randomized_start_x, randomized_start_y)}

            play_interpolated_episode(agent, z_inter, do_eval=True, reset_dict=reset_dict)

            dump_ep = []
            for t_step in agent.episode:
                dump_t = {}
                for key, val in t_step.items():
                    # Convert tensors/numpy arrays to lists for JSON serialization
                    if isinstance(val, torch.Tensor):
                        dump_t[key] = val.detach().numpy().tolist()
                    elif isinstance(val, np.ndarray):
                        dump_t[key] = val.tolist()
                    else:
                        dump_t[key] = val
                dump_ep.append(dump_t)
            episodes_data[str(k)] = dump_ep

        # Save trajectories to JSON
        with open(os.path.join(skill_save_path, "trajectories.json"), 'wt') as f:
            json.dump(episodes_data, f, indent=2)
    
    print("Zero-shot rollouts completed and saved.")


def visualize_interpolated_zero_shot(zero_shot_log_path, exp, skill_idx_1, skill_idx_2, ax=None, **kwargs):
    """
    Visualizes trajectories of interpolated skills with interpolated colors, with specific legend ordering.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        fig = ax.get_figure()

    # Setup plot with maze
    config_subplot(ax, exp=exp, **kwargs)
    exp.learner.agent.env.maze.plot(ax)

    # Infer num_interpolation from subdirectories
    subdirs = [d for d in os.listdir(zero_shot_log_path) if os.path.isdir(os.path.join(zero_shot_log_path, d))]
    if not subdirs:
        print("No subdirectories found in {}.".format(zero_shot_log_path))
        return

    total_skills_to_plot = 0
    for d in subdirs:
        try:
            parts = d.split('-')
            if len(parts) == 2:
                total_skills_to_plot = max(total_skills_to_plot, int(parts[1]))
        except (ValueError, IndexError):
            continue
    
    if total_skills_to_plot < 1:
        print("Could not determine interpolation steps from directory names.")
        return
    num_interpolation = total_skills_to_plot - 1


    # Get base colors
    cmap = plt.get_cmap('tab10') if exp.learner.agent.skill_n <= 10 else plt.get_cmap('tab20')
    color_1_rgb = np.array(cmap(skill_idx_1))
    color_2_rgb = np.array(cmap(skill_idx_2))

    # --- Plotting Loop 1: Trajectories ---
    for i in range(num_interpolation + 2):
        sub_dir_name = "{}-{}".format(i, num_interpolation + 1)
        skill_log_path = os.path.join(zero_shot_log_path, sub_dir_name)
        
        traj_file = os.path.join(skill_log_path, "trajectories.json")
        if not os.path.exists(traj_file):
            print("Warning: trajectories.json not found in {}. Skipping.".format(skill_log_path))
            continue

        with open(traj_file, 'r') as f:
            episodes_data = json.load(f)

        t = i / (num_interpolation + 1)
        inter_color = (1 - t) * color_1_rgb + t * color_2_rgb

        for k_str, trajectory_data in episodes_data.items():
            states_x = [step['state'][0] for step in trajectory_data]
            states_y = [step['state'][1] for step in trajectory_data]
            label = "Skill {}/{} (t={:.2f})".format(i, num_interpolation + 1, t) if k_str == '0' else None
            ax.plot(states_x, states_y, color=inter_color, alpha=0.7, linewidth=2, label=label)

    # --- Plotting Goals for Legend Order ---
    vae = exp.learner.vae
    z1 = exp.learner.agent.skill_embedding(torch.tensor(skill_idx_1))
    z2 = exp.learner.agent.skill_embedding(torch.tensor(skill_idx_2))

    # Plot original centroid 1
    s1_star = vae.get_centroids(dict(skill=torch.tensor(skill_idx_1)))[0].detach().numpy()
    ax.plot(s1_star[0], s1_star[1], 'X', markersize=15, color=color_1_rgb, label="Goal for Skill {}".format(skill_idx_1), zorder=12, markeredgecolor='black')

    # Plot interpolated goals
    for i in range(1, num_interpolation + 1):
        t = i / (num_interpolation + 1)
        inter_color = (1 - t) * color_1_rgb + t * color_2_rgb
        z_inter = (1 - t) * z1 + t * z2
        s_inter_star_normalized = vae.decoder(z_inter.unsqueeze(0)).squeeze(0)
        if vae.normalizes_inputs:
            s_inter_star = vae.normalizer.denormalize(s_inter_star_normalized).detach().numpy()
        else:
            s_inter_star = s_inter_star_normalized.detach().numpy()
        label = "Goal for Skill (t={:.2f})".format(t)
        ax.plot(s_inter_star[0], s_inter_star[1], 'o', markersize=10,
                color=inter_color, label=label, zorder=11, markeredgecolor='black')

    # Plot original centroid 2
    s2_star = vae.get_centroids(dict(skill=torch.tensor(skill_idx_2)))[0].detach().numpy()
    ax.plot(s2_star[0], s2_star[1], 'X', markersize=15, color=color_2_rgb, label="Goal for Skill {}".format(skill_idx_2), zorder=12, markeredgecolor='black')

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    return ax


def calculate_interpolated_skill_L2_error(exp, skill_log_path, skill_latent_vector_z):
    """
    Calculates the L2 distance between the intended goal of a skill and the actual final positions from rollout trajectories.

    Args:
        exp (Experiment): The experiment object, containing the VAE model.
        skill_log_path (str): Path to the directory for a specific skill, containing trajectories.json.
        skill_latent_vector_z (torch.Tensor): The latent vector z for this skill.

    Returns:
        np.ndarray: An array of L2 distances for each trajectory.
    """
    vae = exp.learner.vae
    distances = []

    # 1. Calculate the intended goal state s_star by decoding the latent vector
    s_star_normalized = vae.decoder(skill_latent_vector_z.unsqueeze(0)).squeeze(0)
    if vae.normalizes_inputs:
        s_star = vae.normalizer.denormalize(s_star_normalized).detach().numpy()
    else:
        s_star = s_star_normalized.detach().numpy()

    # 2. Load the trajectory data
    traj_file = os.path.join(skill_log_path, "trajectories.json")
    if not os.path.exists(traj_file):
        print("Warning: trajectories.json not found in {}. Skipping.".format(skill_log_path))
        return np.array([])

    with open(traj_file, 'r') as f:
        episodes_data = json.load(f)

    # 3. For each trajectory, find the final state and calculate the L2 distance
    for k_str, trajectory_data in episodes_data.items():
        if not trajectory_data:
            continue
        # The last state is the final position
        s_final = np.array(trajectory_data[-1]['state'])
        dist = np.linalg.norm(s_star - s_final)
        distances.append(dist)

    return np.array(distances)


def visualize_interpolated_traj_L2_error_bar_plot(zero_shot_log_path, exp, skill_idx_1, skill_idx_2, ax=None, **kwargs):
    """
    Creates a bar plot showing the mean L2 distance error for original and interpolated skills.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # --- 1. Initial Setup (similar to visualize_zero_shot) ---
    subdirs = [d for d in os.listdir(zero_shot_log_path) if os.path.isdir(os.path.join(zero_shot_log_path, d))]
    if not subdirs:
        print("No subdirectories found in {}.".format(zero_shot_log_path))
        return

    total_skills_to_plot = 0
    for d in subdirs:
        try:
            parts = d.split('-')
            if len(parts) == 2:
                total_skills_to_plot = max(total_skills_to_plot, int(parts[1]))
        except (ValueError, IndexError):
            continue

    if total_skills_to_plot < 1:
        print("Could not determine interpolation steps from directory names.")
        return
    num_interpolation = total_skills_to_plot - 1

    cmap = plt.get_cmap('tab10') if exp.learner.agent.skill_n <= 10 else plt.get_cmap('tab20')
    color_1_rgb = np.array(cmap(skill_idx_1))
    color_2_rgb = np.array(cmap(skill_idx_2))

    z1 = exp.learner.agent.skill_embedding(torch.tensor(skill_idx_1))
    z2 = exp.learner.agent.skill_embedding(torch.tensor(skill_idx_2))
    
    # --- 2. Data Collection Loop ---
    mean_errors = []
    std_errors = []
    bar_colors = []
    tick_labels = []

    for i in range(num_interpolation + 2):
        t = i / (num_interpolation + 1)
        
        # Get log path and latent vector for the current skill
        sub_dir_name = "{}-{}".format(i, num_interpolation + 1)
        skill_log_path = os.path.join(zero_shot_log_path, sub_dir_name)
        z_inter = (1 - t) * z1 + t * z2

        # Calculate L2 errors for this skill
        distances = calculate_interpolated_skill_L2_error(exp, skill_log_path, z_inter)

        if distances.size > 0:
            mean_errors.append(np.mean(distances))
            std_errors.append(np.std(distances))
        else:
            mean_errors.append(0)
            std_errors.append(0)

        # Get color and label
        inter_color = (1 - t) * color_1_rgb + t * color_2_rgb
        bar_colors.append(inter_color)
        
        if i == 0:
            tick_labels.append("Skill"+str(skill_idx_1))
        elif i == num_interpolation + 1:
            tick_labels.append("Skill"+str(skill_idx_2))
        else:
            tick_labels.append("t={:.2f}".format(t))

    # --- 3. Plotting ---
    x_pos = np.arange(len(tick_labels))
    ax.bar(x_pos, mean_errors, yerr=std_errors, color=bar_colors, capsize=5, alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_ylabel("Mean L2 Distance to Goal")
    ax.set_title("Skill Trajectory L2 Error")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    
    return ax
