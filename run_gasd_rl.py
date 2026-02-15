# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import sys
import yaml
import json
import subprocess
import shutil
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
    rl_cfg = config['rl']
    maze_type = exp_cfg['maze_type']
    exp_name = exp_cfg['exp_name']
    
    # Priority: CLI argument > config.yaml
    reward_type = args.reward_type or rl_cfg['reward_type']
    
    print("--- Starting GASD RL Training ({0}) for {1} ---".format(reward_type, maze_type))

    # 2. Resolve Experiment Directory
    # We want logs/rl/maze_type/exp_name/reward_type/
    log_root = rl_cfg.get('log_dir', "logs/rl")
    exp_dir = os.path.join(log_root, maze_type, exp_name, reward_type)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # 3. Create JSON config for main.py compatibility
    json_config = {
        "agent_type": "maze",
        "learner_type": "GASD",
        "train_type": "sac_v2",
        "learning_rate": rl_cfg['learning_rate'],
        "batch_size": rl_cfg['batch_size'],
        "buffer_capacity": 1000000,
        "min_buffer_size": 10000,
        "agent_params": {
            "maze_type": maze_type,
            "exp_name": exp_name,
            "reward_type": reward_type,
            "n": 50,
            "pbrs_gamma": rl_cfg['pbrs_gamma'],
            "gamma": rl_cfg['gamma'],
            "hidden_size": rl_cfg['hidden_size'],
            "num_layers": rl_cfg['num_layers'],
            "polyak": rl_cfg['polyak'],
            "normalize_inputs": False,
            "env_reward": False,
            "time_penalty": rl_cfg.get('time_penalty', 0.0),
            "sparse_bonus": rl_cfg.get('sparse_bonus', 0.0),
            "success_threshold": rl_cfg.get('success_threshold', 0.2)
        },
        "save_buffer": False,
        "env_steps_per_cycle": 100,
        "gradient_steps_per_cycle": 50,
        "cycles_per_epoch": 10
    }
    
    # Use a specific name for the config file so main.py doesn't name the experiment 'config'
    temp_json_path = os.path.join(exp_dir, "spectra_config.json")
    with open(temp_json_path, 'w') as f:
        json.dump(json_config, f, indent=4)
    
    # 4. Prepare CLI Command for main.py
    # We set log-dir to the specific exp_dir parent so main.py puts everything inside reward_type folder
    cmd = [
        "python", "main.py",
        "--config-path", temp_json_path,
        "--log-dir", os.path.dirname(exp_dir),
        "--N", str(rl_cfg.get('num_workers', 1)),
        "--dur", str(rl_cfg.get('dur', 500000))
    ]

    print("[Executing] {0}".format(" ".join(cmd)))
    
    # 5. Run Training
    try:
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        process.terminate()
    except Exception as e:
        print("\nError during training: {0}".format(e))
    finally:
        print("\nTraining finished. Logs: {0}".format(exp_dir))

if __name__ == "__main__":
    run_rl_training()