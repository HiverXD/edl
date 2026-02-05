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
import random
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

def collect_oracle_transitions(env, valid_points, grid_size, args):
    """
    Collects transitions using either Grid Sweep or Random Walk.
    """
    unique_raw_transitions = []
    allo_steps = []
    
    get_skeleton = args.get_skeleton
    use_random_walk = args.use_random_walk
    
    # Define available actions
    if get_skeleton:
        actions = [np.array([0, 1.0]), np.array([0, -1.0]), np.array([-1.0, 0]), np.array([1.0, 0])]
        valid_set = set([(round(float(p[0]), 3), round(float(p[1]), 3)) for p in valid_points])
    else:
        actions = [
            np.array([0, grid_size]), np.array([0, -grid_size]),
            np.array([-grid_size, 0]), np.array([grid_size, 0]),
            np.array([grid_size, grid_size]), np.array([-grid_size, grid_size]),
            np.array([grid_size, -grid_size]), np.array([-grid_size, -grid_size])
        ]

    if use_random_walk:
        # Random Walk Mode
        desc = "Generating Random Walks"
        for _ in tqdm(range(args.num_episodes), desc=desc):
            curr_pos = np.array(random.choice(valid_points))
            for t in range(args.random_walk_length):
                action = random.choice(actions)
                
                if get_skeleton:
                    # Discrete Logic
                    target = curr_pos + action
                    target_rounded = (round(float(target[0]), 3), round(float(target[1]), 3))
                    if target_rounded in valid_set:
                        next_pos = target
                    else:
                        next_pos = curr_pos # Wall collision -> stay in place
                else:
                    # Dense Mode: Use Physics
                    env.reset(state=curr_pos)
                    env.step(action)
                    next_pos = env.state.numpy()
                    if env.maze.is_inside_wall(next_pos):
                        next_pos = curr_pos
                
                # Record
                is_done = (t == args.random_walk_length - 1)
                unique_raw_transitions.append((curr_pos.copy(), action, next_pos.copy()))
                
                state_dict_curr = {'xy_agent': curr_pos.copy()}
                state_dict_next = {'xy_agent': next_pos.copy()}
                allo_steps.append(Step(agent_state=state_dict_curr, action=action, episode_done=False))
                if is_done:
                    allo_steps.append(Step(agent_state=state_dict_next, action=None, episode_done=True))
                
                curr_pos = next_pos
    else:
        # Original Grid Sweep Mode
        for start_pos in tqdm(valid_points, desc="Collecting Unique Transitions"):
            for action in actions:
                env.reset(state=start_pos)
                s_curr = env.state.numpy() 
                
                if get_skeleton:
                    target = s_curr + action
                    target_rounded = (round(float(target[0]), 3), round(float(target[1]), 3))
                    if target_rounded in valid_set:
                        s_next = target
                    else:
                        s_next = s_curr # Wall collision
                else:
                    env.step(action)
                    s_next = env.state.numpy()
                    if env.maze.is_inside_wall(s_next):
                        s_next = s_curr
                
                unique_raw_transitions.append((s_curr, action, s_next))
                
                # Simple pair episode for sweep mode
                state_dict_curr = {'xy_agent': s_curr}
                state_dict_next = {'xy_agent': s_next}
                allo_steps.append(Step(agent_state=state_dict_curr, action=action, episode_done=False))
                allo_steps.append(Step(agent_state=state_dict_next, action=None, episode_done=True))
            
    return unique_raw_transitions, allo_steps

def main():
    parser = argparse.ArgumentParser(description="Generate Oracle Transition Data")
    parser.add_argument("--maze_type", type=str, default="square_a")
    parser.add_argument("--grid_size", type=float, default=0.1)
    parser.add_argument("--get_skeleton", action="store_true")
    parser.add_argument("--use_random_walk", action="store_true")
    parser.add_argument("--random_walk_length", type=int, default=50)
    parser.add_argument("--num_episodes", type=int, default=400)
    parser.add_argument("--max_samples_num", type=int, default=10000)
    parser.add_argument("--save_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    print("Generating oracle data for {}...".format(args.maze_type))
    env = Env(n=args.random_walk_length, maze_type=args.maze_type, use_antigoal=False)
    valid_points = get_valid_grid_points(env, args.grid_size, args.get_skeleton)
    
    raw_transitions, allo_steps = collect_oracle_transitions(env, valid_points, args.grid_size, args)
    
    # Repetition logic only for sweep mode if requested (Random walk produces its own volume)
    if not args.use_random_walk:
        repetition = max(1, args.max_samples_num // len(raw_transitions))
        if repetition > 1:
            print("Repeating sweep data {} times.".format(repetition))
            final_raw = []
            final_allo = []
            for _ in range(repetition):
                final_raw.extend(raw_transitions)
                final_allo.extend(allo_steps)
            raw_transitions = final_raw
            allo_steps = final_allo
            save_dir = "data/oracle_transitions/default"
    else:
        save_dir = "data/oracle_transitions/random_walk"

    if args.save_dir is not None:
        save_dir = args.save_dir

    # Save Data
    save_path = os.path.join(save_dir, args.maze_type)
    if not os.path.exists(save_path): os.makedirs(save_path)
    file_path = os.path.join(save_path, "transitions.pkl")
    
    data = {'allo_steps': allo_steps, 'raw_transitions': raw_transitions, 'metadata': vars(args)}
    with open(file_path, 'wb') as f: pickle.dump(data, f)
    print("Data saved to {}. Total transitions: {}".format(file_path, len(raw_transitions)))

if __name__ == "__main__":
    main()
