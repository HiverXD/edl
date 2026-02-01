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
from tqdm import tqdm
from functools import lru_cache
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

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
        """
        x_coords = np.arange(self.min_x, self.max_x, self.resolution)
        y_coords = np.arange(self.min_y, self.max_y, self.resolution)
        
        nodes = []
        coord_to_node = {}
        node_to_coord = []
        
        # Create nodes for all valid grid points
        node_idx = 0
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                if not self.maze.is_inside_wall((x, y)):
                    grid_pos = (j, i)
                    nodes.append(grid_pos)
                    coord_to_node[grid_pos] = node_idx
                    node_to_coord.append((x, y))
                    node_idx += 1
        
        num_nodes = len(nodes)
        adj_matrix = np.zeros((num_nodes, num_nodes))

        # Build adjacency matrix with edge weights as Euclidean distance
        for idx, (jx, iy) in enumerate(tqdm(nodes, desc="Building Maze Graph")):
            # Use 8-directional movement
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = jx + dx, iy + dy
                neighbor_grid_pos = (nx, ny)

                if neighbor_grid_pos in coord_to_node:
                    neighbor_idx = coord_to_node[neighbor_grid_pos]
                    
                    start_coord = node_to_coord[idx]
                    end_coord = node_to_coord[neighbor_idx]

                    # Robust path check: sample points along the line and check all
                    is_path_clear = True
                    num_interp_points = 5 # Number of points to check between nodes
                    for k in range(num_interp_points + 1):
                        alpha = k / num_interp_points
                        interp_point = (start_coord[0] * (1 - alpha) + end_coord[0] * alpha,
                                        start_coord[1] * (1 - alpha) + end_coord[1] * alpha)
                        
                        if self.maze.is_inside_wall(interp_point):
                            is_path_clear = False
                            break
                    
                    if is_path_clear:
                        dist = np.linalg.norm(np.array(start_coord) - np.array(end_coord))
                        adj_matrix[idx, neighbor_idx] = dist

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


def visualize_geodesic_distance_correlation(exp, ax=None, num_samples=100, grid_resolution=0.2):
    """
    Calculates and visualizes the correlation between geodesic distance in state space
    and L2 distance in the latent space.

    Args:
        exp (Experiment): The experiment object.
        ax (matplotlib.axes.Axes, optional): The axes to plot on.
        num_samples (int): The number of random states to sample for the analysis.
        grid_resolution (float): The resolution for discretizing the maze.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 1. Get VAE and Environment
    vae = exp.learner.vae
    env = exp.learner.agent.env

    # 2. Build/Load Maze Graph and All-Pairs Distances
    print("Initializing Geodesic Calculator (builds graph and runs all-pairs shortest path)...")
    geo_calc = GeodesicDistanceCalculator(env.maze, maze_type=env.maze_type, resolution=grid_resolution)
    geodesic_dist_matrix = geo_calc.dist_matrix

    # 3. Sample States and Encode
    print("Sampling {} valid states from the environment...".format(num_samples))
    states = [env.sample() for _ in range(num_samples)]
    state_tensors = torch.tensor(states, dtype=torch.float32)
    
    with torch.no_grad():
        latent_vectors = vae.encoder(state_tensors)

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

    print("Found "+str(len(valid_states))+ "states corresponding to graph nodes.")
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
    # ax.plot([0, 1], [0, 1], 'r--', label='Perfect Correlation (y=x)', zorder=2)
    ax.scatter(d_lat_norm, d_geo_norm, alpha=0.3, edgecolors='none', label='Distance Pairs', zorder=1)
    
    ax.set_xlabel("Normalized Latent Distance")
    ax.set_ylabel("Normalized Geodesic Distance")
    ax.set_title("Geodesic vs. Latent Distance Correlation ({})".format(exp.name))
    ax.text(0.05, 0.95, "Pearson Correlation: {:.3f}".format(corr), transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    
    return ax
