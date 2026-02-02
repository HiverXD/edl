import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc

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
    print("Loading oracle data...")
    data_path = "../data/oracle_transitions/spiral/transitions.pkl"
    
    if not os.path.exists(data_path):
        print("Data file not found.")
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    transitions = data['raw_transitions']
    print("Loaded {} transitions.".format(len(transitions)))
    
    # Create environment for plotting
    env = Env(n=1, maze_type='spiral', use_antigoal=False)
    
    # Plot 1: All transitions as scatter points (start positions)
    fig, ax = plt.subplots(figsize=(10, 10))
    env.maze.plot(ax)
    config_subplot(ax, maze_type='spiral')
    
    # Extract start positions
    starts = np.array([t[0] for t in transitions])
    
    # Check if any are inside walls
    in_wall_indices = [i for i, s in enumerate(starts) if env.maze.is_inside_wall(s)]
    valid_indices = [i for i, s in enumerate(starts) if not env.maze.is_inside_wall(s)]
    
    print("Points inside walls: {}".format(len(in_wall_indices)))
    
    # Plot valid points in blue, invalid in red
    ax.scatter(starts[valid_indices, 0], starts[valid_indices, 1], s=5, c='blue', alpha=0.5, label='Valid Starts')
    if in_wall_indices:
        ax.scatter(starts[in_wall_indices, 0], starts[in_wall_indices, 1], s=20, c='red', marker='x', label='Inside Wall')
        
    ax.set_title("Oracle Sample Distribution (Start Positions)")
    ax.legend()
    plt.savefig("oracle_samples_scatter.png")
    print("Saved oracle_samples_scatter.png")
    
    # Plot 2: Transitions as lines (Vector field)
    fig, ax = plt.subplots(figsize=(12, 12))
    env.maze.plot(ax)
    config_subplot(ax, maze_type='spiral')
    
    # Subsample if too many
    if len(transitions) > 5000:
        indices = np.random.choice(len(transitions), 5000, replace=False)
        plot_transitions = [transitions[i] for i in indices]
    else:
        plot_transitions = transitions
        
    lines = []
    colors = []
    
    for s, a, s_next in plot_transitions:
        lines.append([(s[0], s[1]), (s_next[0], s_next[1])])
        
        # Check if segment intersects wall (simple check: midpoint)
        mid = (s + s_next) / 2
        if env.maze.is_inside_wall(s) or env.maze.is_inside_wall(s_next) or env.maze.is_inside_wall(mid):
            colors.append((1, 0, 0, 1)) # Red
        else:
            colors.append((0, 0, 1, 0.1)) # Transparent Blue
            
    lc = mc.LineCollection(lines, colors=colors, linewidths=1)
    ax.add_collection(lc)
    
    ax.set_title("Oracle Transitions (Blue=Valid, Red=Wall Conflict)")
    plt.savefig("oracle_transitions_lines.png")
    print("Saved oracle_transitions_lines.png")

if __name__ == "__main__":
    visualize_oracle_samples()
