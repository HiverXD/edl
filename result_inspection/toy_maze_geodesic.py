# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import json
import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import lru_cache
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

from .experiment import Experiment
from .toy_maze import config_subplot, ENV_LIMS

class GeodesicDistanceCalculator:
    """
    A class to compute geodesic distances within a maze environment by representing it as a graph.
    """
    def __init__(self, maze, maze_type, resolution=0.1):
        """
        Initializes the calculator by building a graph representation of the maze.
        
        Args:
            maze: An instance of the maze object from the environment.
            maze_type (str): The type of the maze, used for setting boundaries.
            resolution (float): The grid size to discretize the continuous maze space.
        """
        self.maze = maze
        self.maze_type = maze_type
        self.resolution = resolution

        # Determine and store grid boundaries once
        try:
            env_lims = ENV_LIMS[self.maze_type]
            self.min_x, self.max_x = env_lims['x']
            self.min_y, self.max_y = env_lims['y']
        except KeyError:
            self.min_x, self.max_x, self.min_y, self.max_y = -5.5, 5.5, -5.5, 0.5

        self.nodes, self.adj_matrix, self.node_to_coord, self.coord_to_node = self._build_graph()
        
        # Compute all-pairs shortest paths and predecessors
        self.dist_matrix, self.predecessors = dijkstra(
            csgraph=self.adj_matrix, directed=False, return_predecessors=True
        )

    def _get_grid_coords(self, pos):
        """Snaps continuous coordinates to the nearest grid node index (j, i)."""
        j = np.round((pos[0] - self.min_x) / self.resolution).astype(int)
        i = np.round((pos[1] - self.min_y) / self.resolution).astype(int)
        return (j, i)

    @lru_cache(maxsize=None)
    def _build_graph(self):
        """
        Builds a graph representation of the maze.
        Nodes are points on a grid, and edges connect valid neighbors.
        Filters out isolated components (e.g., inside walls) to keep only the main maze.
        """
        x_coords = np.arange(self.min_x, self.max_x, self.resolution)
        y_coords = np.arange(self.min_y, self.max_y, self.resolution)
        
        temp_nodes = []
        temp_coord_to_node = {}
        temp_node_to_coord = []
        epsilon = 1e-4

        # 1. Create initial nodes for all valid grid points
        node_idx = 0
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                if not self.maze.is_inside_wall((x, y)):
                    grid_pos = (j, i)
                    temp_nodes.append(grid_pos)
                    temp_coord_to_node[grid_pos] = node_idx
                    temp_node_to_coord.append((x, y))
                    node_idx += 1
        
        num_temp_nodes = len(temp_nodes)
        temp_adj_matrix = np.zeros((num_temp_nodes, num_temp_nodes))

        # 2. Build initial adjacency matrix
        for idx, (jx, iy) in enumerate(tqdm(temp_nodes, desc="Building Initial Maze Graph")):
            # Use 8-directional movement
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = jx + dx, iy + dy
                neighbor_grid_pos = (nx, ny)

                if neighbor_grid_pos in temp_coord_to_node:
                    neighbor_idx = temp_coord_to_node[neighbor_grid_pos]
                    
                    start_coord = temp_node_to_coord[idx]
                    end_coord = temp_node_to_coord[neighbor_idx]

                    # Robust path check
                    is_path_clear = True
                    num_interp_points = 5 
                    for k in range(num_interp_points + 1):
                        alpha = k / num_interp_points
                        interp_point = (start_coord[0] * (1 - alpha) + end_coord[0] * alpha,
                                        start_coord[1] * (1 - alpha) + end_coord[1] * alpha)
                        
                        if self.maze.is_inside_wall(interp_point):
                            is_path_clear = False
                            break
                    
                    if is_path_clear:
                        dist = np.linalg.norm(np.array(start_coord) - np.array(end_coord))
                        temp_adj_matrix[idx, neighbor_idx] = dist

        # 3. Filter largest connected component
        # Convert to CSR for efficiency
        temp_adj_csr = csr_matrix(temp_adj_matrix)
        n_components, labels = connected_components(csgraph=temp_adj_csr, directed=False, return_labels=True)
        
        # Find the label of the largest component
        counts = np.bincount(labels)
        largest_component_label = np.argmax(counts)
        
        # Select nodes belonging to the largest component
        valid_indices = np.where(labels == largest_component_label)[0]
        
        # Re-build graph structures with only valid nodes
        nodes = []
        coord_to_node = {}
        node_to_coord = []
        
        # Mapping from old index to new index
        old_to_new_idx = {}
        
        for new_idx, old_idx in enumerate(valid_indices):
            grid_pos = temp_nodes[old_idx]
            coord = temp_node_to_coord[old_idx]
            
            nodes.append(grid_pos)
            coord_to_node[grid_pos] = new_idx
            node_to_coord.append(coord)
            old_to_new_idx[old_idx] = new_idx
            
        # Extract sub-matrix for the largest component
        # Efficient slicing using valid_indices
        adj_matrix = temp_adj_matrix[valid_indices][:, valid_indices]
        
        print("Graph Pruning: Kept {} nodes (Largest Component) out of {} initial valid nodes.".format(len(nodes), num_temp_nodes))

        return nodes, csr_matrix(adj_matrix), node_to_coord, coord_to_node

    def get_shortest_path(self, start_pos, end_pos):
        """
        Finds the shortest path between two continuous points in the maze.
        
        Args:
            start_pos (tuple): The (x, y) starting coordinates.
            end_pos (tuple): The (x, y) ending coordinates.
            
        Returns:
            tuple: (distance, path_coordinates)
                   distance (float): The length of the shortest path.
                   path_coordinates (list): A list of (x, y) tuples representing the path.
        """
        start_grid = self._get_grid_coords(start_pos)
        end_grid = self._get_grid_coords(end_pos)
        
        if start_grid not in self.coord_to_node or end_grid not in self.coord_to_node:
            print("Warning: Start or end position is outside the valid maze area.")
            return np.inf, []

        start_node_idx = self.coord_to_node[start_grid]
        end_node_idx = self.coord_to_node[end_grid]
        
        distance = self.dist_matrix[start_node_idx, end_node_idx]
        
        # Reconstruct path from predecessors
        path_indices = []
        curr = end_node_idx
        while curr != start_node_idx and curr != -9999:
            path_indices.append(curr)
            curr = self.predecessors[start_node_idx, curr]
        if curr == -9999:
            print("Warning: No path found between the points.")
            return np.inf, []
        path_indices.append(start_node_idx)
        path_indices.reverse()
        
        path_coordinates = [self.node_to_coord[i] for i in path_indices]
        
        return distance, path_coordinates


def visualize_single_geodesic_path(exp, start_pos, end_pos, ax=None, resolution=0.1):
    """
    Calculates and visualizes the geodesic shortest path between two points in a maze.
    
    Args:
        exp (Experiment): The experiment object.
        start_pos (tuple): The (x, y) starting coordinates.
        end_pos (tuple): The (x, y) ending coordinates.
        ax (matplotlib.axes.Axes, optional): The axes to plot on.
        resolution (float): The grid resolution for the pathfinding graph.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    env = exp.learner.agent.env
    config_subplot(ax, exp=exp)
    env.maze.plot(ax)

    # Initialize calculator and get path
    print("Building graph for geodesic distance calculation (may take a moment)...")
    geo_calculator = GeodesicDistanceCalculator(env.maze, maze_type=env.maze_type, resolution=resolution)
    distance, path = geo_calculator.get_shortest_path(start_pos, end_pos)
    print("Geodesic distance: {:.2f}".format(distance))

    # Plot path if found
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'r-', linewidth=2.5, label='Shortest Path', zorder=10)
        ax.plot(start_pos[0], start_pos[1], 'bo', markersize=10, label='Start', zorder=11)
        ax.plot(end_pos[0], end_pos[1], 'go', markersize=10, label='End', zorder=11)
        ax.grid()
        ax.legend()

    ax.set_title("Shortest Path from {} to {}".format(start_pos, end_pos))
    
    return ax


def calculate_latent_distance(exp, s1, s2):
    """
    Calculates the L2 distance between the latent representations of two states.

    Args:
        exp (Experiment): The experiment object, containing the VAE model.
        s1 (tuple): The (x, y) coordinates of the first state.
        s2 (tuple): The (x, y) coordinates of the second state.
    
    Returns:
        float: The L2 distance between the latent vectors z1 and z2.
    """
    vae = exp.learner.vae
    
    # Convert states to tensors
    s1_tensor = torch.tensor(s1, dtype=torch.float32).unsqueeze(0)
    s2_tensor = torch.tensor(s2, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        z1 = vae.encoder(s1_tensor)
        z2 = vae.encoder(s2_tensor)

    distance = torch.norm(z1 - z2, p=2).item()
    
    print("State 1: {} -> Latent z1".format(s1))
    print("State 2: {} -> Latent z2".format(s2))
    print("L2 Distance between z1 and z2: {:.4f}".format(distance))
    
    return distance

class LaplacianEncoderWrapper(nn.Module):
    def __init__(self, model_path):
        super(LaplacianEncoderWrapper, self).__init__()
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        args = checkpoint['args']
        self.mean = torch.from_numpy(checkpoint['mean']).float()
        self.std = torch.from_numpy(checkpoint['std']).float()
        
        # Reconstruct architecture
        from geometry_aware_skill_discovery.train_laplacian_encoder import LaplacianEncoder
        self.encoder = LaplacianEncoder(input_dim=2, hidden_dim=args.hidden_dim, output_dim=args.dim)
        self.encoder.load_state_dict(state_dict)
        self.encoder.eval()
        
    def forward(self, x):
        # Apply normalization
        x_norm = (x - self.mean) / self.std
        return self.encoder(x_norm)

def load_laplacian_encoder(maze_type, experiment_name="default"):
    root_dir = os.environ.get("ROOT_DIR", ".")
    model_path = os.path.join(root_dir, "logs/laplacian_encoder", maze_type, experiment_name, "model.pth.tar")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found at {}".format(model_path))
    return LaplacianEncoderWrapper(model_path)

def visualize_geodesic_distance_correlation(exp, encoder=None, ax=None, num_samples=100, grid_resolution=0.2, title_suffix=""):
    """
    Calculates and visualizes the correlation between geodesic distance in state space
    and distance in a latent space (VAE or Laplacian).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 1. Get Encoder and Environment
    # If encoder is not provided, use the VAE encoder from the experiment
    if encoder is None:
        raw_encoder = exp.learner.vae.encoder
        # Simple wrapper for VAE encoder to handle potential normalization if needed
        # (Though EDL VAE usually handles it internally or doesn't use it the same way)
        def encode_fn(x):
            return raw_encoder(x)
    else:
        encode_fn = encoder

    env = exp.learner.agent.env

    # 2. Build/Load Maze Graph and All-Pairs Distances
    print("Initializing Geodesic Calculator...")
    geo_calc = GeodesicDistanceCalculator(env.maze, maze_type=env.maze_type, resolution=grid_resolution)
    geodesic_dist_matrix = geo_calc.dist_matrix

    # 3. Sample States and Encode
    print("Sampling {} valid states from the environment...".format(num_samples))
    states = [env.sample() for _ in range(num_samples)]
    state_tensors = torch.tensor(states, dtype=torch.float32)
    
    with torch.no_grad():
        latent_vectors = encode_fn(state_tensors)

    # Find corresponding node indices for sampled states
    node_indices = []
    valid_states = []
    valid_latents = []
    for i, state in enumerate(states):
        grid_coords = geo_calc._get_grid_coords(state)
        if grid_coords in geo_calc.coord_to_node:
            node_indices.append(geo_calc.coord_to_node[grid_coords])
            valid_states.append(state)
            valid_latents.append(latent_vectors[i])

    print("Found {} states corresponding to graph nodes.".format(len(valid_states)))
    if len(valid_states) < 2:
        print("Not enough valid states to compute correlation.")
        return

    # 4. Calculate Paired Distances
    latent_distances = []
    geodesic_distances = []
    
    num_valid_samples = len(valid_states)
    for i in tqdm(range(num_valid_samples), desc="Calculating paired distances"):
        for j in range(i + 1, num_valid_samples):
            # Latent distance
            d_latent = torch.norm(valid_latents[i] - valid_latents[j], p=2)
            
            # Geodesic distance
            node_i = node_indices[i]
            node_j = node_indices[j]
            d_geodesic = geodesic_dist_matrix[node_i, node_j]

            # Ignore pairs with no path
            if not np.isinf(d_geodesic):
                latent_distances.append(d_latent.item())
                geodesic_distances.append(d_geodesic)


    # 5. Normalize the Distances
    d_lat_arr = np.array(latent_distances)
    d_geo_arr = np.array(geodesic_distances)

    d_lat_norm = (d_lat_arr - d_lat_arr.min()) / (d_lat_arr.max() - d_lat_arr.min())
    d_geo_norm = (d_geo_arr - d_geo_arr.min()) / (d_geo_arr.max() - d_geo_arr.min())

    # 6. Calculate Pearson Correlation
    corr = np.corrcoef(d_lat_norm, d_geo_norm)[0, 1]

    # 7. Plotting
    ax.scatter(d_lat_norm, d_geo_norm, alpha=0.3, edgecolors='none', label='Distance Pairs', zorder=1)
    # Perfect correlation line
    ax.plot([0, 1], [0, 1], 'r--', label='y=x (Ideal)', zorder=2)
    
    ax.set_xlabel("Normalized Latent Distance")
    ax.set_ylabel("Normalized Geodesic Distance")
    title = "Geodesic vs. Latent Distance Correlation {}".format(title_suffix)
    ax.set_title(title)
    ax.text(0.05, 0.95, "Pearson Correlation: {:.3f}".format(corr), transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    
    return ax

def visualize_distance_heatmap(exp, metric_fn, s0=(0.0, 0.0), ax=None, resolution=0.1, title="Distance Heatmap"):
    """
    Visualizes the distance from a reference state s0 to all other states in the maze
    using a provided metric function.

    Args:
        exp (Experiment): The experiment object.
        metric_fn (callable): Function f(states, s0) -> distances. 
                              states is (N, 2) numpy array/tensor, s0 is (2,) tuple/tensor.
                              Should return (N,) array of distances.
        s0 (tuple): The reference (source) state coordinates (x, y).
        ax (matplotlib.axes.Axes, optional): The axes to plot on.
        resolution (float): Grid resolution for the heatmap.
        title (str): Title of the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    env = exp.learner.agent.env
    
    # 1. Setup Grid
    try:
        env_lims = ENV_LIMS[env.maze_type]
        min_x, max_x = env_lims['x']
        min_y, max_y = env_lims['y']
    except KeyError:
        min_x, max_x, min_y, max_y = -5.5, 5.5, -5.5, 0.5

    x_coords = np.arange(min_x, max_x, resolution)
    y_coords = np.arange(min_y, max_y, resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Flatten for validity check and metric calculation
    flat_X = X.flatten()
    flat_Y = Y.flatten()
    grid_points = np.stack([flat_X, flat_Y], axis=1) # (N, 2)

    # 2. Check Validity (Wall Mask)
    valid_mask = np.array([not env.maze.is_inside_wall((p[0], p[1])) for p in grid_points])
    
    # 3. Compute Distances using metric_fn
    # Filter valid points to pass to metric_fn (to avoid potential errors with invalid states)
    valid_points = grid_points[valid_mask]
    
    # Convert s0 to appropriate format if needed inside metric_fn, but here passing as tuple/array
    # metric_fn is expected to handle batch computation or iteration
    if len(valid_points) > 0:
        dists = metric_fn(valid_points, s0)
    else:
        dists = np.array([])

    # 4. Reconstruct Heatmap
    heatmap_data = np.full(X.shape, np.nan) # Fill with NaNs
    
    # We need to map back valid_points indices to original grid indices
    # Since we flattened and masked, we can just fill in order if we iterate or use indices
    # It's easier to use the mask directly on the flattened array
    flat_dists = np.full(flat_X.shape, np.nan)
    flat_dists[valid_mask] = dists
    heatmap_data = flat_dists.reshape(X.shape)

    # 5. Plot
    config_subplot(ax, exp=exp)
    env.maze.plot(ax)
    
    # Plot heatmap
    # Use 'viridis' or 'plasma' for distances. 'inf' or NaN will be transparent/white usually.
    mesh = ax.pcolormesh(X, Y, heatmap_data, cmap='viridis', shading='auto', alpha=0.8)
    
    # Mark s0
    ax.plot(s0[0], s0[1], 'r*', markersize=15, markeredgecolor='black', label='s0', zorder=10)
    
    ax.set_title(title)
    # Add colorbar
    plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    
    return ax
