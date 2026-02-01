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
from dist_train.workers.utils import ReplayBuffer
from .experiment import Experiment, EXPERIMENT_DIR
from agents.maze_agents.toy_maze.skill_discovery.edl import VQVAEDiscriminator


NUM_TRAJECTORIES = 20
TRAJECTORY_KWARGS = dict(alpha=0.2, linewidth=2)

SAVEFIG_KWARGS = dict(bbox_inches='tight', transparent=True)

ENV_LIMS = dict(
    square_a=dict(xlim=(-0.55, 4.55), ylim=(-4.55, 0.55), x=(-0.5, 4.5), y=(-4.5, 0.5)),
    square_bottleneck=dict(xlim=(-0.55, 9.55), ylim=(-0.55, 9.55), x=(-0.5, 9.5), y=(-0.5, 9.5)),
    square_corridor=dict(xlim=(-5.55, 5.55), ylim=(-0.55, 0.55), x=(-5.5, 5.5), y=(-0.5, 0.5)),
    square_corridor2=dict(xlim=(-5.55, 5.55), ylim=(-0.55, 0.55), x=(-5.5, 5.5), y=(-0.5, 0.5)),
    square_tree=dict(xlim=(-6.55, 6.55), ylim=(-6.55, 0.55), x=(-6.5, 6.5), y=(-6.5, 0.5))
)


def load_exp_data(exp_name, notebook_mode=True):
    exp = Experiment(exp_name, notebook_mode=notebook_mode)
    agent = exp.learner.agent
    if agent.skill_n <= 10:
        cmap = plt.get_cmap('tab10')
    elif 10 < agent.skill_n <= 20:
        cmap = plt.get_cmap('tab20')
    else:
        cmap = plt.get_cmap('viridis', agent.skill_n)
    return exp, cmap


def config_subplot(ax, maze_type=None, title=None, extra_lim=0., fontsize=14, exp=None):
    if maze_type is None and exp is not None:
        maze_type = exp.learner.agent.env.maze_type

    if maze_type is not None:
        env_config = ENV_LIMS[maze_type]
        ax.set_xlim(env_config["xlim"][0] - extra_lim, env_config["xlim"][1] + extra_lim)
        ax.set_ylim(env_config["ylim"][0] - extra_lim, env_config["ylim"][1] + extra_lim)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for p in ["left", "right", "top", "bottom"]:
        ax.spines[p].set_visible(False)


def play_episode(agent, skill, do_eval, reset_dict={}):
    agent.reset(**reset_dict)
    agent.curr_skill = agent.curr_skill * 0 + skill
    while not agent.env.is_done:
        agent.step(do_eval)


def play_interpolated_episode(agent, z_interpolated, do_eval=True, reset_dict={}):
    """
    Plays an episode using an interpolated skill vector.
    Temporarily overrides the agent's preprocess_skill to directly use the provided vector.
    """
    original_preprocess = agent.preprocess_skill
    # Override preprocess_skill to just pass the vector through
    # Ensure skill_vec is a tensor before detach()
    agent.preprocess_skill = lambda skill_vec: skill_vec.detach() if isinstance(skill_vec, torch.Tensor) else skill_vec

    agent.reset(skill=z_interpolated, **reset_dict)
    while not agent.env.is_done:
        agent.step(do_eval)

    # Restore original method
    agent.preprocess_skill = original_preprocess


def _plot_all_skills(exp, cmap, ax=None, reset_dict=None, start_pos_base=(0.0, 0.0), random_range=0.2, alpha=1., linewidth=1.):
    agent = exp.learner.agent
    agent.env.maze.plot(ax)

    # If a specific reset_dict is provided, use it. Otherwise, randomize.
    final_reset_dict = reset_dict
    if final_reset_dict is None:
        # Use a randomized starting state within a square if no specific start is given
        randomized_start_x = start_pos_base[0] + np.random.uniform(-random_range, random_range)
        randomized_start_y = start_pos_base[1] + np.random.uniform(-random_range, random_range)
        final_reset_dict = {'state': (randomized_start_x, randomized_start_y)}

    for skill_idx in range(agent.skill_n):
        # Collect rollout
        play_episode(agent, skill_idx, do_eval=False, reset_dict=final_reset_dict)
        # Plot trajectory
        ax.plot(*agent.rollout, label="Skill #{}".format(skill_idx), color=cmap(skill_idx), alpha=alpha,
                linewidth=linewidth, zorder=10)
    # Mark initial state with a dot
    ax.plot(agent.rollout[0][0], agent.rollout[1][0], marker='o', markersize=8, color='black', zorder=11)


def plot_all_skills(exp, cmap, ax=None, reset_dict=None, start_pos_base=(0.0, 0.0), random_range=0.2, notebook_mode=True, desc=None, figsize=(5, 5), **kwargs):
    desc = desc or "Trajectories"
    tqdm_ = tqdm_notebook if notebook_mode else tqdm

    if ax is None:
        return_ax = True
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        return_ax = False

    for _ in tqdm_(range(NUM_TRAJECTORIES), desc=desc, disable=False, leave=True, total=NUM_TRAJECTORIES):
        # Pass all relevant parameters to _plot_all_skills
        _plot_all_skills(exp, cmap, ax, reset_dict=reset_dict, start_pos_base=start_pos_base, random_range=random_range, **TRAJECTORY_KWARGS)

    config_subplot(ax, exp=exp, **kwargs)

    if return_ax:
        return ax


def load_smm_buffer(exp_name, epoch=None, notebook_mode=True):
    exp, _ = load_exp_data(exp_name, notebook_mode=notebook_mode)

    valid_epochs = sorted([int(d.split("_")[0]) for d in os.listdir(exp.exp_dir) if "replay_buffer" in d])

    if epoch is None:
        epoch = valid_epochs[-1]

    assert epoch in valid_epochs, "Replay buffer for epoch {} not found. Found epochs: {}".format(epoch, valid_epochs)

    # Load buffer
    config = exp.get_config()
    config['load_buffer'] = True
    config['buffer_path'] = os.path.join(exp.exp_dir, "{:04d}_replay_buffer".format(epoch))
    buffer = ReplayBuffer(None, config, verbose_load=True)

    # Get all training samples from the buffer
    batch_size = buffer.batch_size
    buffer.batch_size = int(buffer.size)
    dataset = buffer.make_batch(normalize=False)['next_state']
    buffer.batch_size = batch_size

    return exp, dataset


def visualize_smm_samples(exp_name, epoch=None, ax=None, sample_frac=1., figsize=(5, 5), **kwargs):
    exp, dataset = load_smm_buffer(exp_name, epoch)
    env = exp.learner.agent.env

    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)

    env.maze.plot(ax)
    config_subplot(ax, exp=exp, **kwargs)

    num_samples = int(sample_frac * dataset.shape[0])
    _ = ax.scatter(dataset[:num_samples, 0], dataset[:num_samples, 1], s=3, marker='o')


def load_vqvae(exp_name, verbose=False):
    exp_dir = os.path.join(EXPERIMENT_DIR, exp_name)

    # Load config
    config = json.load(open(os.path.join(exp_dir, "config.json")))
    if verbose:
        print(json.dumps(config, indent=2))

    # Load model
    model = VQVAEDiscriminator(state_size=2, **config['vae_args'])
    model.load_state_dict(torch.load(os.path.join(exp_dir, "model.pth.tar")))
    model.eval()

    # Load train loss
    loss = json.loads(json.load(open(os.path.join(exp_dir, "loss.json"))))

    return model, config, loss


def state_coverage(exp, cell_size, ax=None, notebook_mode=True, **kwargs):
    """
    Computes and visualizes the state coverage of an agent's learned skills.
    - Divides the maze into a grid of cell_size.
    - Runs the agent for all skills to collect trajectories.
    - Counts how many times any part of a trajectory falls into each grid cell.
    - Plots a heatmap of the visit counts.
    - Prints the quantitative state coverage percentage.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    agent = exp.learner.agent
    env = agent.env

    # 1. Get Maze Boundaries and Create Grid
    try:
        env_lims = ENV_LIMS[env.maze_type]
        min_x, max_x = env_lims['x']
        min_y, max_y = env_lims['y']
    except KeyError:
        print("Warning: Maze type '{}' not in ENV_LIMS. Using default limits.".format(env.maze_type))
        min_x, max_x, min_y, max_y = -5.5, 5.5, -5.5, 0.5

    n_cells_x = math.ceil((max_x - min_x) / cell_size)
    n_cells_y = math.ceil((max_y - min_y) / cell_size)
    visit_counts = np.zeros((n_cells_y, n_cells_x))

    # 2. Collect Trajectories and Populate Visit Counts
    num_trajectories_per_skill = 5  # Collect a few trajectories for each skill for robustness
    all_trajectories = []

    for skill_idx in range(agent.skill_n):
        for _ in range(num_trajectories_per_skill):
            play_episode(agent, skill_idx, do_eval=True)
            # agent.rollout is a tuple of (x_coords, y_coords)
            trajectory_states = np.stack(agent.rollout, axis=-1)
            all_trajectories.append(trajectory_states)

    tqdm_ = tqdm_notebook if notebook_mode else tqdm
    for trajectory in tqdm_(all_trajectories):
        for state in trajectory:
            # Convert state (x, y) to grid indices (i, j)
            x, y = state[0], state[1]
            j = int((x - min_x) / cell_size)
            i = int((y - min_y) / cell_size)

            # Ensure indices are within bounds
            if 0 <= i < n_cells_y and 0 <= j < n_cells_x:
                visit_counts[i, j] += 1

    # 3. Calculate and Print Quantitative Metrics
    # For total_valid_cells, check if the center of each cell is valid using the direct wall check
    total_valid_cells = 0
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cell_center_y = min_y + (i + 0.5) * cell_size
            cell_center_x = min_x + (j + 0.5) * cell_size
            cell_center_coord = (cell_center_x, cell_center_y)

            if not env.maze.is_inside_wall(cell_center_coord):
                total_valid_cells += 1

    
    visited_cells = np.count_nonzero(visit_counts)
    if total_valid_cells > 0:
        coverage_percent = (visited_cells / total_valid_cells) * 100
    else:
        coverage_percent = 0.0

    print("Grid Size: {}x{}".format(n_cells_x, n_cells_y))
    print("Visited Cells: {} in {}".format(visited_cells, total_valid_cells))
    print("State Coverage: {:.2f}%".format(coverage_percent))


    # 4. Visualize Heatmap
    config_subplot(ax, exp=exp, **kwargs)
    env.maze.plot(ax)

    # Use a log scale for better visualization of low-count cells, adding 1 to avoid log(0)
    log_counts = np.log1p(visit_counts)

    im = ax.imshow(log_counts, origin='lower',
                   extent=[min_x, max_x, min_y, max_y],
                   aspect='auto', alpha=0.7, cmap='viridis')
    
    # Add a color bar
    fig = ax.get_figure()
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Log(Visit Count + 1)")
    
    return ax

def rollout_zero_shot(exp, skill_idx_1, skill_idx_2, num_interpolation, num_rollout_traj_each_mode, base_save_dir="zero_shot_logs", start_pos_base=(0.0, -0.5), random_range=0.2):
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


def visualize_zero_shot(zero_shot_log_path, exp, skill_idx_1, skill_idx_2, ax=None, **kwargs):
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
    
    # The number of interpolation steps is the total number of skills minus the two base skills.
    # The total number of directories is num_interpolation + 2.
    # The directory format is {i}-{total_dirs-1}, so total_skills_to_plot is actually the max index, which is num_interpolation+1
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


def calculate_skill_L2_error(exp, skill_log_path, skill_latent_vector_z):
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


def visualize_traj_L2_error_bar_plot(zero_shot_log_path, exp, skill_idx_1, skill_idx_2, ax=None, **kwargs):
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
        distances = calculate_skill_L2_error(exp, skill_log_path, z_inter)

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
