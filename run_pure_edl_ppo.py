# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import json
import yaml
import subprocess
from argparse import ArgumentParser

def run_pure_edl_ppo():
    parser = ArgumentParser(description="Run Pure Original EDL Baseline (PPO).")
    parser.add_argument("--maze_type", type=str, default="square_corridor")
    parser.add_argument("--vae_logdir", type=str, default="logs/vqvae/square_corridor/curriculum/identity")
    args = parser.parse_args()

    vae_abs_path = os.path.abspath(args.vae_logdir)
    
    agent_params = {
        "vae_logdir": vae_abs_path,
        "env_reward": False,
        "hidden_size": 128,
        "num_layers": 4,
        "normalize_inputs": False,
        "horizon": 2048,
        "mini_batch_size": 64,
        "clip_range": 0.2,
        "entropy_lambda": 0.01,
        "gae_lambda": 0.98,
        "env_params": {
            "maze_type": args.maze_type,
            "n": 50,
            "done_on_success": True
        }
    }

    json_config = {
        "agent_type": "maze",
        "learner_type": "EDL",
        "train_type": "ppo",
        "learning_rate": 0.0003,
        "batch_size": 256,
        "rollouts_per_cycle": 40,
        "update_epochs_per_rollout": 10, # Standard PPO setting
        "agent_params": agent_params
    }

    log_root = os.path.abspath("logs/rl")
    maze_log_dir = os.path.join(log_root, args.maze_type)
    if not os.path.exists(maze_log_dir):
        os.makedirs(maze_log_dir)

    temp_json_path = os.path.join(maze_log_dir, "pure_edl_ppo.json")
    with open(temp_json_path, 'w') as f:
        json.dump(json_config, f, indent=4)

    cmd = [
        "python", "main.py",
        "--config-path", temp_json_path,
        "--log-dir", maze_log_dir,
        "--N", "1",
        "--dur", "50"
    ]

    print("[Executing Pure EDL PPO] {0}".format(" ".join(cmd)))
    try:
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
    finally:
        print("\nFinished Pure EDL PPO Training.")

if __name__ == "__main__":
    run_pure_edl_ppo()