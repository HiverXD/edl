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

