# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import sys
import yaml
import shutil
import subprocess
from argparse import ArgumentParser

def run_command(command):
    print("[Executing] {0}".format(" ".join(command)))
    result = subprocess.call(command)
    if result != 0:
        print("[Error] Command failed with exit code {0}".format(result))
        sys.exit(result)

def main():
    parser = ArgumentParser(description="Meta Orchestrator for Curriculum Laplacian Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to curriculum config file")
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    exp_cfg = config['experiment']
    maze_type = exp_cfg['maze_type']
    exp_name = exp_cfg['exp_name']
    dim = exp_cfg['dim']
    hidden_dim = exp_cfg.get('hidden_dim', 256)

    print("--- Starting Curriculum Training for {0} ---".format(maze_type))

    prev_log_dir = None

    for i, stage in enumerate(config['stages']):
        stage_id = "stage_{0}".format(stage['id'])
        print(">>> Processing Stage {0}: {1}".format(stage['id'], stage.get('name', '')))

        # --- Path Definitions ---
        # Data: data/oracle_transitions/curriculum/{maze_type}/stage_n/maze_type/transitions.pkl
        data_dir = os.path.join("data/oracle_transitions", exp_name, maze_type, stage_id)
        data_path = os.path.join(data_dir, maze_type, "transitions.pkl")
        
        # Log: logs/laplacian_encoder/{maze_type}/curriculum/stage_n/
        current_exp_id = os.path.join(exp_name, stage_id)
        log_dir = os.path.join("logs/laplacian_encoder", maze_type, current_exp_id)

        # 2. Data Generation Task
        gen_cfg = stage['generation']
        gen_cmd = [
            "python", "geometry_aware_skill_discovery/generate_laplacian_oracle.py",
            "--maze_type", maze_type,
            "--grid_size", str(gen_cfg['grid_size']),
            "--save_dir", data_dir
        ]
        if gen_cfg.get('get_skeleton'): gen_cmd.append("--get_skeleton")
        if gen_cfg.get('use_random_walk'): gen_cmd.append("--use_random_walk")
        if gen_cfg.get('num_episodes'): 
            gen_cmd.extend(["--num_episodes", str(gen_cfg['num_episodes'])])
        
        run_command(gen_cmd)

        # 3. Model Handover (Resume Logic)
        train_cfg = stage['training']
        if train_cfg.get('resume') and prev_log_dir:
            print("[Handover] Copying model from {0} to {1} for resume...".format(prev_log_dir, log_dir))
            if not os.path.exists(log_dir): os.makedirs(log_dir)
            
            # Copy model.pth.tar and training_stats.json
            for filename in ["model.pth.tar", "training_stats.json"]:
                src = os.path.join(prev_log_dir, filename)
                dst = os.path.join(log_dir, filename)
                if os.path.exists(src):
                    shutil.copy2(src, dst)

        # 4. Training Task
        train_cmd = [
            "python", "geometry_aware_skill_discovery/train_laplacian_encoder.py",
            "--maze_type", maze_type,
            "--data_path", data_path,
            "--exp_name", current_exp_id,
            "--dim", str(dim),
            "--hidden_dim", str(hidden_dim),
            "--lr", str(train_cfg['lr'])
        ]
        
        if train_cfg.get('resume'):
            train_cmd.extend([
                "--resume",
                "--additional_epochs", str(train_cfg.get('additional_epochs', 100))
            ])
        else:
            train_cmd.extend(["--epochs", str(train_cfg.get('epochs', 100))])
            
        run_command(train_cmd)

        # Update previous log dir for next stage handover
        prev_log_dir = log_dir

    # 5. Finalize: Copy config.yaml to the final log folder for documentation
    if prev_log_dir:
        dst_config = os.path.join(prev_log_dir, "curriculum_config.yaml")
        shutil.copy2(args.config, dst_config)
        print("[Done] Curriculum training complete. Config saved to {0}".format(dst_config))

if __name__ == "__main__":
    main()
