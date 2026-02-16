# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import sys
import yaml
import json
import subprocess
from argparse import ArgumentParser

def run_rl_training():
    parser = ArgumentParser(description="Run GASD RL Training from config.yaml")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--reward_type", type=str, default=None, help="Override reward type (static/dynamic)")
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    exp_cfg = config['experiment']
    rl_main = config['rl']
    common = rl_main['common']
    
    maze_type = exp_cfg['maze_type']
    exp_name = exp_cfg['exp_name']
    
    reward_type = args.reward_type or rl_main.get('reward_type', 'dynamic')
    specific = rl_main[reward_type]
    
    print("--- Starting GASD RL Training ({0}) for {1} ---".format(reward_type, maze_type))

    # 2. Resolve Base Log Directory
    # We want logs/rl/maze_type/exp_name/
    log_root = os.path.abspath(common.get('log_dir', "logs/rl"))
    base_exp_dir = os.path.join(log_root, maze_type, exp_name)
    if not os.path.exists(base_exp_dir):
        os.makedirs(base_exp_dir)

    # 3. Create JSON config
    agent_params = {
        "maze_type": maze_type,
        "exp_name": exp_name,
        "reward_type": reward_type,
        "n": 50,
        "gamma": common['gamma'],
        "hidden_size": common['hidden_size'],
        "num_layers": common['num_layers'],
        "polyak": common['polyak'],
        "target_entropy": common.get('target_entropy', None), # Add this
        "normalize_inputs": False,
        "env_reward": False,
        "sparse_bonus": common.get('sparse_bonus', 0.0),
        "success_threshold": common.get('success_threshold', 0.2),
        "gaussian_bonus": common.get('gaussian_bonus', 0.0),
        "gaussian_std": common.get('gaussian_std', 0.5),
        "reward_scale": specific.get('reward_scale', 1.0),
        "time_penalty": specific.get('time_penalty', 0.0),
        "pbrs_gamma": specific.get('pbrs_gamma', 0.99)
    }

    json_config = {
        "agent_type": "maze",
        "learner_type": "GASD",
        "train_type": "sac_v2",
        "learning_rate": common['learning_rate'],
        "batch_size": common['batch_size'],
        "buffer_capacity": 1000000,
        "min_buffer_size": 10000,
        "cycles_per_epoch": 10,
        "env_steps_per_cycle": 100,
        "gradient_steps_per_cycle": 50,
        "logging_keys": common.get('logging_keys', []), # Pass logging keys
        "agent_params": agent_params
    }
    
    # Save the config file in the base_exp_dir
    # The filename (static.json or dynamic.json) will be the experiment ID
    temp_json_path = os.path.join(base_exp_dir, "{0}.json".format(reward_type))
    with open(temp_json_path, 'w') as f:
        json.dump(json_config, f, indent=4)
    
    # 4. Prepare CLI Command for main.py
    # main.py will see config at .../base_dir/static.json
    # It will create a folder .../base_dir/static/ and put config.json there.
    cmd = [
        "python", "main.py",
        "--config-path", temp_json_path,
        "--log-dir", base_exp_dir,
        "--N", str(common.get('num_workers', 1)),
        "--dur", str(common.get('dur', 500))
    ]

    print("[Executing] {0}".format(" ".join(cmd)))
    
    try:
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
    finally:
        print("\nFinished RL Session.")

if __name__ == "__main__":
    run_rl_training()