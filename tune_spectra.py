# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import optuna
import yaml
import json
import subprocess
import numpy as np
import copy
from argparse import ArgumentParser

def objective(trial, maze_type, reward_type, base_config):
    # 1. Suggest Parameters
    # Using suggest_loguniform and suggest_uniform (Standard for 1.5.0)
    reward_scale = trial.suggest_loguniform("reward_scale", 0.1, 10.0)
    time_penalty = trial.suggest_loguniform("time_penalty", 0.001, 0.1)
    target_entropy = trial.suggest_uniform("target_entropy", -2.0, -0.1)
    
    # 2. Create Temporary Config using deepcopy to avoid contamination
    config = copy.deepcopy(base_config)
    config['experiment']['maze_type'] = maze_type
    config['rl']['common']['target_entropy'] = target_entropy
    config['rl'][reward_type]['reward_scale'] = reward_scale
    config['rl'][reward_type]['time_penalty'] = time_penalty
    
    if "huber" in reward_type:
        config['rl'][reward_type]['huber_delta'] = 1.0
        
    tmp_config_path = "configs/optuna_{0}_{1}_{2}.yaml".format(maze_type, reward_type, trial.number)
    with open(tmp_config_path, 'w') as f:
        yaml.dump(config, f)

    seed = 42 + trial.number
    stats_path = os.path.join("logs/rl", maze_type, "curriculum", reward_type, str(seed), "training_stats.json")
    if os.path.exists(stats_path):
        import shutil
        shutil.rmtree(os.path.dirname(stats_path), ignore_errors=True)

    # 3. Run Training
    cmd = ["python", "train_gasd_sac.py", "--config", tmp_config_path, "--reward_type", reward_type, "--seed", str(seed)]
    
    try:
        # We use a timeout or check_call. Since we have early stopping, this should be fine.
        subprocess.check_call(cmd)
    except Exception as e:
        print("[Trial {0}] Failed with error: {1}".format(trial.number, e))
        return 0.0
    finally:
        if os.path.exists(tmp_config_path): os.remove(tmp_config_path)

    # 4. Score Calculation
    if not os.path.exists(stats_path): return 0.0
    with open(stats_path, 'r') as f:
        try: history = json.load(f)
        except: return 0.0
    
    if not history: return 0.0
    
    last_window = history[-100:] if len(history) >= 100 else history
    final_succ = np.mean([h['success'] for h in last_window])
    
    first_success_step = len(history)
    moving_avg = 0
    for i, h in enumerate(history):
        moving_avg = 0.9 * moving_avg + 0.1 * h['success']
        if moving_avg >= 0.8:
            first_success_step = i
            break
    
    speed_bonus = 1.0 - (first_success_step / (len(history) + 1e-8))
    total_score = float(final_succ + speed_bonus)
    
    print("[Trial {0}] Final Score: {1:.4f} (Succ: {2:.2f}, Speed: {3:.2f})".format(
        trial.number, total_score, final_succ, speed_bonus))
    
    return total_score

def main():
    parser = ArgumentParser()
    parser.add_argument("--maze_type", type=str, default="square_ant_maze_1")
    parser.add_argument("--reward_type", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=15)
    args = parser.parse_args()

    with open("config.yaml", 'r') as f: base_config = yaml.load(f)

    # Setup Study with RandomSampler to avoid Scipy/Numpy conflicts in Python 3.5
    study_name = "spectra_{0}_{1}".format(args.maze_type, args.reward_type)
    log_root = "logs/optuna"
    if not os.path.exists(log_root): os.makedirs(log_root)
    storage_path = "sqlite:///{0}/tuning.db".format(os.path.abspath(log_root))
    
    # [FIX] Force RandomSampler for legacy environment stability
    sampler = optuna.samplers.RandomSampler(seed=42)
    
    study = optuna.create_study(
        study_name=study_name, storage=storage_path,
        direction="maximize", load_if_exists=True,
        sampler=sampler
    )

    print("--- Starting SPECTRA Tuning (Robust Random Search) ---")
    study.optimize(lambda trial: objective(trial, args.maze_type, args.reward_type, base_config), 
                   n_trials=args.n_trials)

    print("\nBest Params:", study.best_params)

if __name__ == "__main__":
    main()
