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
        """
        self.maze = maze
        self.maze_type = maze_type
        self.resolution = resolution

        # Use limits from shared constant
        try:
            env_lims = ENV_LIMS[self.maze_type]
            self.min_x, self.max_x = env_lims['x']
            self.min_y, self.max_y = env_lims['y']
        except KeyError:
            self.min_x, self.max_x, self.min_y, self.max_y = -5.5, 5.5, -5.5, 0.5

        self.nodes, self.adj_matrix, self.node_to_coord, self.coord_to_node = self._build_graph()
        
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
        """
        x_coords = np.arange(self.min_x, self.max_x, self.resolution)
        y_coords = np.arange(self.min_y, self.max_y, self.resolution)
        
        temp_nodes = []
        temp_coord_to_node = {}
        temp_node_to_coord = []
        epsilon = 1e-4

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

        for idx, (jx, iy) in enumerate(tqdm(temp_nodes, desc="Building Initial Maze Graph")):
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = jx + dx, iy + dy
                neighbor_grid_pos = (nx, ny)

                if neighbor_grid_pos in temp_coord_to_node:
                    neighbor_idx = temp_coord_to_node[neighbor_grid_pos]
                    start_coord = temp_node_to_coord[idx]
                    end_coord = temp_node_to_coord[neighbor_idx]

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

        temp_adj_csr = csr_matrix(temp_adj_matrix)
        n_components, labels = connected_components(csgraph=temp_adj_csr, directed=False, return_labels=True)
        counts = np.bincount(labels)
        largest_component_label = np.argmax(counts)
        valid_indices = np.where(labels == largest_component_label)[0]
        
        nodes = []
        coord_to_node = {}
        node_to_coord = []
        for new_idx, old_idx in enumerate(valid_indices):
            grid_pos = temp_nodes[old_idx]
            coord = temp_node_to_coord[old_idx]
            nodes.append(grid_pos)
            coord_to_node[grid_pos] = new_idx
            node_to_coord.append(coord)
            
        adj_matrix = temp_adj_matrix[valid_indices][:, valid_indices]
        print("Graph Pruning: Kept {} nodes (Largest Component) out of {} initial valid nodes.".format(len(nodes), num_temp_nodes))

        return nodes, csr_matrix(adj_matrix), node_to_coord, coord_to_node

    def get_shortest_path(self, start_pos, end_pos):
        """Finds the shortest path between two continuous points."""
        start_grid = self._get_grid_coords(start_pos)
        end_grid = self._get_grid_coords(end_pos)
        
        if start_grid not in self.coord_to_node or end_grid not in self.coord_to_node:
            print("Warning: Start or end position is outside the valid maze area.")
            return np.inf, []

        start_node_idx = self.coord_to_node[start_grid]
        end_node_idx = self.coord_to_node[end_grid]
        distance = self.dist_matrix[start_node_idx, end_node_idx]
        
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
        return distance, [self.node_to_coord[i] for i in path_indices]


def visualize_single_geodesic_path(exp, start_pos, end_pos, ax=None, resolution=0.1):
    """Calculates and visualizes the geodesic shortest path."""
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    env = exp.learner.agent.env
    config_subplot(ax, exp=exp)
    env.maze.plot(ax)
    geo_calculator = GeodesicDistanceCalculator(env.maze, maze_type=env.maze_type, resolution=resolution)
    distance, path = geo_calculator.get_shortest_path(start_pos, end_pos)
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'r-', linewidth=2.5, label='Shortest Path', zorder=10)
        ax.plot(start_pos[0], start_pos[1], 'bo', markersize=10, label='Start')
        ax.plot(end_pos[0], end_pos[1], 'go', markersize=10, label='End')
        ax.grid(); ax.legend()
    ax.set_title("Shortest Path: {:.2f}".format(distance))
    return ax

class LaplacianEncoderWrapper(nn.Module):
    def __init__(self, model_path):
        super(LaplacianEncoderWrapper, self).__init__()
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        args = checkpoint['args']
        self.mean = torch.from_numpy(checkpoint['mean']).float()
        self.std = torch.from_numpy(checkpoint['std']).float()
        from geometry_aware_skill_discovery.train_laplacian_encoder import LaplacianEncoder
        self.encoder = LaplacianEncoder(input_dim=2, hidden_dim=args.hidden_dim, output_dim=args.dim)
        self.encoder.load_state_dict(state_dict); self.encoder.eval()
    def forward(self, x):
        return self.encoder((x - self.mean) / self.std)

def load_laplacian_encoder(maze_type, experiment_name="default"):
    root_dir = os.environ.get("ROOT_DIR", ".")
    model_path = os.path.join(root_dir, "logs/laplacian_encoder", maze_type, experiment_name, "model.pth.tar")
    if not os.path.exists(model_path): raise FileNotFoundError("Model not found at {}".format(model_path))
    return LaplacianEncoderWrapper(model_path)

def visualize_geodesic_distance_correlation(exp, encoder=None, ax=None, num_samples=100, grid_resolution=0.2, title_suffix=""):
    """Visualizes correlation between geodesic and latent distances."""
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    env = exp.learner.agent.env
    encode_fn = encoder if encoder is not None else exp.learner.vae.encoder
    geo_calc = GeodesicDistanceCalculator(env.maze, maze_type=env.maze_type, resolution=grid_resolution)
    states = [env.sample() for _ in range(num_samples)]
    with torch.no_grad(): latent_vectors = encode_fn(torch.tensor(states, dtype=torch.float32))
    node_indices, valid_states, valid_latents = [], [], []
    for i, state in enumerate(states):
        grid_coords = geo_calc._get_grid_coords(state)
        if grid_coords in geo_calc.coord_to_node:
            node_indices.append(geo_calc.coord_to_node[grid_coords]); valid_states.append(state); valid_latents.append(latent_vectors[i])
    if len(valid_states) < 2: return
    latent_distances, geodesic_distances = [], []
    for i in range(len(valid_states)):
        for j in range(i + 1, len(valid_states)):
            d_geodesic = geo_calc.dist_matrix[node_indices[i], node_indices[j]]
            if not np.isinf(d_geodesic):
                # Calculate L2 distance safely for both Tensors and Numpy arrays
                z_i = valid_latents[i].detach().cpu().numpy() if torch.is_tensor(valid_latents[i]) else valid_latents[i]
                z_j = valid_latents[j].detach().cpu().numpy() if torch.is_tensor(valid_latents[j]) else valid_latents[j]
                d_latent = np.linalg.norm(z_i - z_j)
                
                latent_distances.append(d_latent)
                geodesic_distances.append(d_geodesic)
    d_lat_arr, d_geo_arr = np.array(latent_distances), np.array(geodesic_distances)
    d_lat_norm = (d_lat_arr - d_lat_arr.min()) / (d_lat_arr.max() - d_lat_arr.min())
    d_geo_norm = (d_geo_arr - d_geo_arr.min()) / (d_geo_arr.max() - d_geo_arr.min())
    corr = np.corrcoef(d_lat_norm, d_geo_norm)[0, 1]
    ax.scatter(d_lat_norm, d_geo_norm, alpha=0.3); ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlabel("Normalized Latent Distance"); ax.set_ylabel("Normalized Geodesic Distance")
    ax.set_title("Corr: {:.3f} {}".format(corr, title_suffix))
    return ax

def visualize_distance_heatmap(exp, metric_fn, s0=(0.0, 0.0), ax=None, resolution=0.1, title="Distance Heatmap"):
    """Visualizes distance from reference state s0 to all other states."""
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    env = exp.learner.agent.env
    try:
        env_lims = ENV_LIMS[env.maze_type]
        min_x, max_x = env_lims['x']
        min_y, max_y = env_lims['y']
    except KeyError:
        raise Exception('key error, add toy_maze.py, ENV_LIMS')
    x_coords = np.arange(min_x, max_x, resolution)
    y_coords = np.arange(min_y, max_y, resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
    valid_mask = np.array([not env.maze.is_inside_wall(p) for p in grid_points])
    valid_points = grid_points[valid_mask]
    if len(valid_points) > 0: dists = metric_fn(valid_points, s0)
    else: dists = np.array([])
    flat_dists = np.full(X.size, np.nan); flat_dists[valid_mask] = dists
    config_subplot(ax, maze_type=env.maze_type); env.maze.plot(ax)
    mesh = ax.pcolormesh(X, Y, flat_dists.reshape(X.shape), cmap='viridis', shading='auto', alpha=0.8)
    ax.plot(s0[0], s0[1], 'r*', markersize=15); ax.set_title(title)
    ax.set_xlim(min_x - 0.5, max_x + 0.5)
    ax.set_ylim(min_y - 0.5, max_y + 0.5)
    plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    return ax