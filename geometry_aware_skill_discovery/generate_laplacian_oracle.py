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

# Define Step namedtuple directly
Step = collections.namedtuple('Step', 'agent_state, action, episode_done')

def get_valid_grid_points(env, resolution=0.1, get_skeleton=False):
    """
    Generates a grid of valid points or exact segment centers.
    """
    if get_skeleton:
        print("Skeleton mode: Using exact corridor segment centers.")
        return list(env.maze._locs)

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

def collect_oracle_transitions(env, valid_points, grid_size, get_skeleton=False):
    """
    Collects transitions using Grid Sweep or Random Walk (for skeleton).
    """
    import random
    unique_raw_transitions = []
    
    if get_skeleton:
        # Skeleton Random Walk Mode
        actions = [np.array([0, 1.0]), np.array([0, -1.0]), np.array([-1.0, 0]), np.array([1.0, 0])]
        valid_set = set([(round(float(p[0]), 3), round(float(p[1]), 3)) for p in valid_points])
        
        ep_length = 20
        for start_pos in tqdm(valid_points, desc="Generating Skeleton Random Walks"):
            curr_pos = np.array(start_pos)
            for _ in range(ep_length):
                action = actions[random.randint(0, 3)]
                target = curr_pos + action
                target_rounded = (round(float(target[0]), 3), round(float(target[1]), 3))
                
                if target_rounded in valid_set:
                    next_pos = target
                else:
                    next_pos = curr_pos # Wall collision -> stay in place (Self-loop)
                
                unique_raw_transitions.append((curr_pos, action, next_pos))
                curr_pos = next_pos
    else:
        # Dense Mode: Grid Sweep Logic
        actions = [
            np.array([0, grid_size]), np.array([0, -grid_size]),
            np.array([-grid_size, 0]), np.array([grid_size, 0]),
            np.array([grid_size, grid_size]), np.array([-grid_size, grid_size]),
            np.array([grid_size, -grid_size]), np.array([-grid_size, -grid_size])
        ]

        for start_pos in tqdm(valid_points, desc="Collecting Unique Transitions"):
            for action in actions:
                env.reset(state=start_pos)
                s_curr = env.state.numpy() 
                
                env.step(action)
                s_next = env.state.numpy()
                if not env.maze.is_inside_wall(s_next):
                    unique_raw_transitions.append((s_curr, action, s_next))
            
    return unique_raw_transitions

def main():
    parser = argparse.ArgumentParser(description="Generate Oracle Transition Data")
    parser.add_argument("--maze_type", type=str, default="square_a")
    parser.add_argument("--grid_size", type=float, default=0.1)
    parser.add_argument("--get_skeleton", action="store_true")
    parser.add_argument("--max_samples_num", type=int, default=10000)
    parser.add_argument("--save_dir", type=str, default="data/oracle_transitions")
    
    args = parser.parse_args()
    
    print("Generating oracle data for {}...".format(args.maze_type))
    env = Env(n=1, maze_type=args.maze_type, use_antigoal=False)
    valid_points = get_valid_grid_points(env, args.grid_size, args.get_skeleton)
    unique_raw = collect_oracle_transitions(env, valid_points, args.grid_size, args.get_skeleton)
    
    repetition = max(1, args.max_samples_num // len(unique_raw))
    final_raw = []
    for _ in range(repetition):
        final_raw.extend(unique_raw)
    
    allo_steps = []
    for s_curr, action, s_next in final_raw:
        state_dict_curr = {'xy_agent': s_curr}
        state_dict_next = {'xy_agent': s_next}
        step_start = Step(agent_state=state_dict_curr, action=action, episode_done=False)
        step_end = Step(agent_state=state_dict_next, action=None, episode_done=True)
        allo_steps.append(step_start)
        allo_steps.append(step_end)
    
    save_path = os.path.join(args.save_dir, args.maze_type)
    if not os.path.exists(save_path): os.makedirs(save_path)
    file_path = os.path.join(save_path, "transitions.pkl")
    
    data = {'allo_steps': allo_steps, 'raw_transitions': final_raw, 'metadata': vars(args)}
    with open(file_path, 'wb') as f: pickle.dump(data, f)
    print("Data saved to {}. Total transitions: {}".format(file_path, len(final_raw)))

if __name__ == "__main__":
    main()