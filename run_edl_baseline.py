# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import json
import yaml
import subprocess
from argparse import ArgumentParser

def run_edl_baseline():
    parser = ArgumentParser(description="Run Original EDL Baseline Training using SAC v1.")
    parser.add_argument("--maze_type", type=str, default="square_corridor")
    parser.add_argument("--vae_logdir", type=str, default="logs/vqvae/square_corridor/curriculum/identity")
    args = parser.parse_args()

    vae_abs_path = os.path.abspath(args.vae_logdir)
    if not os.path.exists(vae_abs_path):
        raise FileNotFoundError("VAE logdir not found: {0}".format(vae_abs_path))

    agent_params = {
        "vae_logdir": vae_abs_path,
        "env_reward": False,
        "hidden_size": 128,
        "num_layers": 4,
        "normalize_inputs": False,
        "alpha": 0.1,
        "polyak": 0.95,
        "env_params": {
            "maze_type": args.maze_type,
            "n": 50,
            "done_on_success": True
        }
    }

    json_config = {
        "agent_type": "maze",
        "learner_type": "EDL",
        "train_type": "sac",
        "learning_rate": 0.0003,
        "batch_size": 256,
        "buffer_capacity": 1000000,
        "min_buffer_size": 10000,
        "cycles_per_epoch": 10,
        "env_steps_per_cycle": 100,
        "gradient_steps_per_cycle": 50,
        "agent_params": agent_params
    }

    log_root = os.path.abspath("logs/rl")
    maze_log_dir = os.path.join(log_root, args.maze_type)
    if not os.path.exists(maze_log_dir):
        os.makedirs(maze_log_dir)

    temp_json_path = os.path.join(maze_log_dir, "baseline_edl_sac.json")
    with open(temp_json_path, 'w') as f:
        json.dump(json_config, f, indent=4)

    cmd = [
        "python", "main.py",
        "--config-path", temp_json_path,
        "--log-dir", maze_log_dir,
        "--N", "1",
        "--dur", "50"
    ]

    print("[Executing Baseline SAC v1] {0}".format(" ".join(cmd)))
    try:
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
    finally:
        print("\nFinished Baseline SAC v1 Training.")

if __name__ == "__main__":
    run_edl_baseline()