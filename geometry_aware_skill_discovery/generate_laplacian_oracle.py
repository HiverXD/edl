# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import sys
import os
import argparse
import pickle
import numpy as np
import collections
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from agents.maze_agents.toy_maze.env.maze_env import Env
from result_inspection.toy_maze import ENV_LIMS

# Define Step namedtuple directly to avoid importing laplacian_dual_dynamics (f-string issues in Python 3.5)
Step = collections.namedtuple('Step', 'agent_state, action, episode_done')

def get_valid_grid_points(env, resolution=0.1):
    """
    Generates a grid of valid points within the maze boundaries.
    """
    try:
        env_lims = ENV_LIMS[env.maze_type]
        min_x, max_x = env_lims['x']
        min_y, max_y = env_lims['y']
    except KeyError:
        print("Warning: Maze type '{}' not in ENV_LIMS. Using default limits.".format(env.maze_type))
        min_x, max_x, min_y, max_y = -5.5, 5.5, -5.5, 0.5

    x_coords = np.arange(min_x, max_x, resolution)
    y_coords = np.arange(min_y, max_y, resolution)
    
    valid_points = []
    
    for y in y_coords:
        for x in x_coords:
            if not env.maze.is_inside_wall((x, y)):
                valid_points.append((x, y))
                
    return valid_points

def collect_oracle_transitions(env, valid_points, grid_size):
    """
    Collects transitions from each valid point in 4 cardinal directions.
    """
    allo_steps = []
    raw_transitions = []
    
    # 8 directions: Up, Down, Left, Right + Diagonals
    actions = [
        np.array([0, grid_size]),  # Up
        np.array([0, -grid_size]), # Down
        np.array([-grid_size, 0]), # Left
        np.array([grid_size, 0]),  # Right
        np.array([grid_size, grid_size]),   # Up-Right
        np.array([-grid_size, grid_size]),  # Up-Left
        np.array([grid_size, -grid_size]),  # Down-Right
        np.array([-grid_size, -grid_size])  # Down-Left
    ]
    
    for start_pos in tqdm(valid_points, desc="Collecting Transitions"):
        for action in actions:
            env.reset(state=start_pos)
            s_curr = env.state.numpy() 
            
            env.step(action)
            s_next = env.state.numpy()
            
            state_dict_curr = {'xy_agent': s_curr}
            state_dict_next = {'xy_agent': s_next}
            
            step_start = Step(agent_state=state_dict_curr, action=action, episode_done=False)
            step_end = Step(agent_state=state_dict_next, action=None, episode_done=True)
            
            allo_steps.append(step_start)
            allo_steps.append(step_end)
            
            raw_transitions.append((s_curr, action, s_next))
            
    return allo_steps, raw_transitions

def main():
    parser = argparse.ArgumentParser(description="Generate Oracle Transition Data for Laplacian Representation Learning")
    parser.add_argument("--maze_type", type=str, default="square_a", help="Type of maze environment")
    parser.add_argument("--grid_size", type=float, default=0.1, help="Grid resolution for sampling")
    parser.add_argument("--save_dir", type=str, default="data/oracle_transitions", help="Directory to save the dataset")
    
    args = parser.parse_args()
    
    print("Generating oracle data for {} with grid size {}...".format(args.maze_type, args.grid_size))
    
    env = Env(n=1, maze_type=args.maze_type, use_antigoal=False)
    
    print("Identifying valid grid points...")
    valid_points = get_valid_grid_points(env, args.grid_size)
    print("Found {} valid points.".format(len(valid_points)))
    
    print("Collecting transitions...")
    allo_steps, raw_transitions = collect_oracle_transitions(env, valid_points, args.grid_size)
    print("Collected {} transitions.".format(len(raw_transitions)))
    
    save_path = os.path.join(args.save_dir, args.maze_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, "transitions.pkl")
    
    data = {
        'allo_steps': allo_steps,
        'raw_transitions': raw_transitions,
        'metadata': {
            'maze_type': args.maze_type,
            'grid_size': args.grid_size,
            'num_transitions': len(raw_transitions)
        }
    }
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
        
    print("Data saved to {}".format(file_path))

if __name__ == "__main__":
    main()
