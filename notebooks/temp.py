import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

# 경로 설정: notebooks 디렉토리에서 실행된다고 가정하고 상위 디렉토리 추가
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# 필요한 모듈 임포트
# (주의: 이 스크립트는 프로젝트 루트가 아닌 notebooks 폴더에서 실행될 것을 가정)
# 만약 루트에서 실행한다면 경로 조정 필요. 여기서는 상대경로 .. 을 sys.path에 추가했으므로
# 루트 기준의 import가 가능해야 함.

try:
    from agents.maze_agents.toy_maze.env.maze_env import Env
    from result_inspection.toy_maze_geodesic import GeodesicDistanceCalculator
    from result_inspection.toy_maze import ENV_LIMS
    from geometry_aware_skill_discovery.generate_laplacian_oracle import Step # Pickle 로드용
except ImportError as e:
    print("Import Error: {}. Make sure you are running from the 'notebooks' directory or adjust sys.path.".format(e))
    sys.exit(1)

def debug_oracle_data():
    print("\n--- 1. Oracle Data Debugging ---")
    data_path = "../data/oracle_transitions/square_a/transitions.pkl"
    if not os.path.exists(data_path):
        print("Data file not found at {}".format(data_path))
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    transitions = data['raw_transitions']
    print("Total transitions: {}".format(len(transitions)))
    
    env = Env(n=1, maze_type='square_a', use_antigoal=False)
    maze = env.maze
    
    # Check a sample for wall collisions
    invalid_starts = 0
    invalid_ends = 0
    wall_crossings = 0 # Simple check: midpoint in wall
    
    sample_size = min(1000, len(transitions))
    indices = np.random.choice(len(transitions), sample_size, replace=False)
    
    for idx in indices:
        s, a, s_next = transitions[idx]
        
        if maze.is_inside_wall(s):
            invalid_starts += 1
        if maze.is_inside_wall(s_next):
            invalid_ends += 1
            
        # Midpoint check
        mid = (s + s_next) / 2
        if maze.is_inside_wall(mid):
            wall_crossings += 1
            
    print("Sampled {} transitions:".format(sample_size))
    print("  Invalid Start Positions (Inside Wall): {}".format(invalid_starts))
    print("  Invalid End Positions (Inside Wall): {}".format(invalid_ends))
    print("  Wall Crossings (Midpoint in Wall): {}".format(wall_crossings))

    # Plotting some transitions
    fig, ax = plt.subplots(figsize=(6, 6))
    maze.plot(ax)
    
    # Plot invalid transitions in red
    count = 0
    for idx in indices:
        s, a, s_next = transitions[idx]
        mid = (s + s_next) / 2
        if maze.is_inside_wall(s) or maze.is_inside_wall(s_next) or maze.is_inside_wall(mid):
            ax.plot([s[0], s_next[0]], [s[1], s_next[1]], 'r-', alpha=0.5)
            count += 1
            if count > 50: break # Don't clutter too much
            
    ax.set_title("Red lines: Transitions interacting with walls (Sampled)")
    plt.savefig("debug_oracle_transitions.png")
    print("Saved plot to debug_oracle_transitions.png")

def debug_validity_logic():
    print("\n--- 2. Validity Logic Debugging ---")
    env = Env(n=1, maze_type='square_a', use_antigoal=False)
    maze = env.maze
    resolution = 0.1
    
    print("Building Geodesic Graph...")
    geo_calc = GeodesicDistanceCalculator(maze, 'square_a', resolution=resolution)
    
    # Grid setup similar to visualize_distance_heatmap
    env_lims = ENV_LIMS['square_a']
    min_x, max_x = env_lims['x']
    min_y, max_y = env_lims['y']
    
    x_coords = np.arange(min_x, max_x, resolution)
    y_coords = np.arange(min_y, max_y, resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    # Compare methods
    is_wall_valid_count = 0
    in_graph_count = 0
    
    # Debugging discrepancies
    discrepancy_points = []

    for p in grid_points:
        # Method 1: is_inside_wall (False means valid/empty space)
        is_valid_by_wall_check = not maze.is_inside_wall(p)
        if is_valid_by_wall_check:
            is_wall_valid_count += 1
            
        # Method 2: Geodesic Graph
        p_grid = geo_calc._get_grid_coords(p)
        is_valid_by_graph = p_grid in geo_calc.coord_to_node
        
        if is_valid_by_graph:
            in_graph_count += 1
            
        if is_valid_by_wall_check != is_valid_by_graph:
            discrepancy_points.append((p, is_valid_by_wall_check, is_valid_by_graph))

    print("Total Grid Points: {}".format(len(grid_points)))
    print("Valid points by is_inside_wall: {}".format(is_wall_valid_count))
    print("Valid points in Geodesic Graph: {}".format(in_graph_count))
    
    if len(discrepancy_points) > 0:
        print("DISCREPANCY FOUND! {} points differ.".format(len(discrepancy_points)))
        print("Sample discrepancies (Point, is_inside_wall_valid, in_graph_valid):")
        for i in range(min(10, len(discrepancy_points))):
            print("  {}".format(discrepancy_points[i]))
            
        # Plot discrepancies
        fig, ax = plt.subplots(figsize=(6, 6))
        maze.plot(ax)
        
        # Plot points valid in wall check but not in graph (Blue x)
        valid_wall_only = [p for p, w, g in discrepancy_points if w and not g]
        if valid_wall_only:
            vw_x, vw_y = zip(*valid_wall_only)
            ax.plot(vw_x, vw_y, 'bx', label='Valid in Wall Check Only')
            
        # Plot points valid in graph but not in wall check (Red o) - Should ideally be 0
        valid_graph_only = [p for p, w, g in discrepancy_points if g and not w]
        if valid_graph_only:
            vg_x, vg_y = zip(*valid_graph_only)
            ax.plot(vg_x, vg_y, 'ro', label='Valid in Graph Only')
            
        ax.legend()
        ax.set_title("Validity Logic Discrepancies")
        plt.savefig("debug_validity_discrepancy.png")
        print("Saved plot to debug_validity_discrepancy.png")

if __name__ == "__main__":
    debug_oracle_data()
    debug_validity_logic()
