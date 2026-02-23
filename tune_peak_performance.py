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

def get_next_seed(base_path):
    """ Dynamically find the next available folder index (seed) """
    if not os.path.exists(base_path):
        return 42 # Default starting seed
    
    existing_dirs = [d for d in os.listdir(base_path) if d.isdigit()]
    if not existing_dirs:
        return 42
    
    return max([int(d) for d in existing_dirs]) + 1

def objective(trial, args, base_config):
    # 1. Define Discrete Grids (To avoid Scipy bugs)
    entropy_grid = [round(x, 2) for x in np.linspace(-2.0, 0.0, 11).tolist()]
    
    if args.reward_type == 'static_huber':
        scale_grid = [round(x, 2) for x in np.linspace(0.5, 4.0, 15).tolist()]
        pen_grid = [round(x, 4) for x in np.linspace(0.0005, 0.02, 15).tolist()]
    elif args.reward_type == 'dynamic_huber':
        scale_grid = [round(x, 2) for x in np.linspace(1.0, 8.0, 15).tolist()]
        pen_grid = [round(x, 3) for x in np.linspace(0.01, 0.15, 15).tolist()]
    elif args.reward_type == 'static':
        scale_grid = [round(x, 3) for x in np.linspace(0.05, 1.0, 15).tolist()]
        pen_grid = [round(x, 4) for x in np.linspace(0.0005, 0.02, 15).tolist()]
    else: # Dynamic (Vanilla)
        scale_grid = [round(x, 2) for x in np.linspace(1.0, 8.0, 15).tolist()]
        pen_grid = [round(x, 3) for x in np.linspace(0.01, 0.15, 15).tolist()]

    ent_idx = trial.suggest_int('ent_idx', 0, len(entropy_grid) - 1)
    sc_idx = trial.suggest_int('sc_idx', 0, len(scale_grid) - 1)
    pen_idx = trial.suggest_int('pen_idx', 0, len(pen_grid) - 1)
    
    target_entropy = entropy_grid[ent_idx]
    reward_scale = scale_grid[sc_idx]
    time_penalty = pen_grid[pen_idx]

    # 2. Dynamic Seed Determination
    # We look at existing folders to ensure we never overwrite or resume incorrectly
    base_log_dir = os.path.join("logs/rl", args.maze_type, "curriculum", args.reward_type)
    trial_seed = get_next_seed(base_log_dir)
    
    # Store the actual seed in trial attributes for future reference/inspection
    trial.set_user_attr("actual_seed", int(trial_seed))

    # 3. Create Temporary Config
    config = copy.deepcopy(base_config)
    config['experiment']['maze_type'] = args.maze_type
    config['rl']['common']['target_entropy'] = float(target_entropy)
    config['rl'][args.reward_type]['reward_scale'] = float(reward_scale)
    config['rl'][args.reward_type]['time_penalty'] = float(time_penalty)
    
    tmp_config_path = "configs/peak_optuna_{0}_{1}.yaml".format(args.reward_type, trial.number)
    with open(tmp_config_path, 'w') as f:
        yaml.dump(config, f)

    # 4. Run Training via Subprocess
    # We use trial_seed calculated dynamically
    cmd = ["python", "train_gasd_sac.py", "--config", tmp_config_path, "--reward_type", args.reward_type, "--seed", str(trial_seed)]
    
    try:
        subprocess.check_call(cmd)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print("[Trial {0}] Training Failed: {1}".format(trial.number, e))
        return 0.0
    finally:
        if os.path.exists(tmp_config_path): os.remove(tmp_config_path)

    # 5. Score Calculation
    stats_path = os.path.join(base_log_dir, str(trial_seed), "training_stats.json")
    if not os.path.exists(stats_path): return 0.0
    with open(stats_path, 'r') as f:
        try: history = json.load(f)
        except: return 0.0
    if not history: return 0.0
    
    final_succ = np.mean([h['success'] for h in history[-100:]]) if len(history) >= 100 else 0.0
    first_success_step = len(history)
    moving_avg = 0
    for i, h in enumerate(history):
        moving_avg = 0.9 * moving_avg + 0.1 * h['success']
        if moving_avg >= 0.8:
            first_success_step = i
            break
    speed_bonus = 1.0 - (first_success_step / (len(history) + 1e-8))
    
    total_score = float(final_succ + speed_bonus)
    print("[Trial {0}] Peak Result -> Score: {1:.4f} (Seed: {2}, Scale: {3}, Pen: {4})".format(
        trial.number, total_score, trial_seed, reward_scale, time_penalty))
    return total_score

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--maze_type', type=str, default='square_ant_maze_1')
    parser.add_argument('--reward_type', type=str, required=True)
    parser.add_argument('--n_trials', type=int, default=20)
    args = parser.parse_args()

    with open("config.yaml", 'r') as f: base_config = yaml.load(f)

    # Use a clean study name for the final Bayesian peak search
    study_name = "peak_final_{0}_{1}".format(args.maze_type, args.reward_type)
    storage_path = "sqlite:///logs/optuna/tuning.db"
    
    sampler = optuna.samplers.TPESampler(n_startup_trials=5)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction='maximize',
        sampler=sampler,
        load_if_exists=True
    )

    print("\n>>> Starting DYNAMIC SEED BAYESIAN SEARCH for {0} | Mode: {1}".format(args.maze_type, args.reward_type))
    print(">>> New logs will start from seed index: {0}".format(get_next_seed(os.path.join("logs/rl", args.maze_type, "curriculum", args.reward_type))))
    
    study.optimize(lambda trial: objective(trial, args, base_config), n_trials=args.n_trials)

    print("\nBest Peak Performance Found:")
    print("  Score: {0:.4f}".format(study.best_value))
    print("  Best Params Indices: {0}".format(study.best_params))
