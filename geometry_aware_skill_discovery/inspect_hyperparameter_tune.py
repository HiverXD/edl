# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import optuna
import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser

def inspect_tuning():
    parser = ArgumentParser(description="Inspect Optuna HPO progress with detailed metrics.")
    parser.add_argument("--db_path", type=str, default="logs/optuna/tuning.db")
    parser.add_argument("--maze", type=str, default="square_ant_maze_1", help="Filter by maze type")
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print("[Error] Database not found.")
        return

    storage_url = "sqlite:///{0}".format(os.path.abspath(args.db_path))
    try:
        summaries = optuna.get_all_study_summaries(storage=storage_url)
    except:
        print("[Error] Could not read DB.")
        return

    # Filter studies by maze name
    target_studies = [s for s in summaries if args.maze in s.study_name]

    print("\n" + "="*95)
    print(" SPECTRA HPO REPORT FOR: {0}".format(args.maze.upper()))
    print("="*95)

    if not target_studies:
        print("No studies found for maze: {0}".format(args.maze))
        return

    for summary in target_studies:
        study_name = summary.study_name
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        df = study.trials_dataframe()
        completed = df[df['state'] == 'COMPLETE']
        
        print("\n>>> Reward Type: {0}".format(study_name.split(args.maze + "_")[-1].upper()))
        print("-" * 95)
        
        results = []
        for _, row in completed.iterrows():
            trial_num = int(row['number'])
            score = row['value']
            if score is None or score <= 0.0: continue # Filter out garbage trials

            # Dynamic Parameter Resolution (Legacy vs V4)
            # 1. Grids from tune_peak_performance.py
            ent_grid = [round(x, 2) for x in np.linspace(-2.0, 0.0, 11).tolist()]
            if "huber" in study_name:
                sc_grid = [round(x, 2) for x in np.linspace(0.5, 4.0, 15).tolist()] if "static" in study_name else [round(x, 2) for x in np.linspace(1.0, 8.0, 15).tolist()]
                pn_grid = [round(x, 4) for x in np.linspace(0.0005, 0.02, 15).tolist()] if "static" in study_name else [round(x, 3) for x in np.linspace(0.01, 0.15, 15).tolist()]
            else:
                sc_grid = [round(x, 3) for x in np.linspace(0.05, 1.0, 15).tolist()] if "static" in study_name else [round(x, 2) for x in np.linspace(1.0, 8.0, 15).tolist()]
                pn_grid = [round(x, 4) for x in np.linspace(0.0005, 0.02, 15).tolist()] if "static" in study_name else [round(x, 3) for x in np.linspace(0.01, 0.15, 15).tolist()]

            # 2. Extract Values
            def get_val(legacy_key, idx_key, grid):
                if 'params_' + legacy_key in row and not pd.isnull(row['params_' + legacy_key]):
                    return row['params_' + legacy_key]
                if 'params_' + idx_key in row and not pd.isnull(row['params_' + idx_key]):
                    return grid[int(row['params_' + idx_key])]
                return 0.0

            scale = get_val('reward_scale', 'sc_idx', sc_grid)
            penalty = get_val('time_penalty', 'pen_idx', pn_grid)
            entropy = get_val('target_entropy', 'ent_idx', ent_grid)

            # Derive reward type for path
            r_type = study_name.split(args.maze + "_")[-1]
            if "peak_v4" in study_name: # Handle study name variants
                r_type = study_name.split("peak_v4_" + args.maze + "_")[-1]
            
            seed = 42 + trial_num
            stats_path = os.path.join("logs/rl", args.maze, "curriculum", r_type, str(seed), "training_stats.json")
            
            final_succ, conv_step = 0.0, "N/A"
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    try: history = json.load(f)
                    except: history = []
                if history:
                    last_window = history[-100:]
                    final_succ = np.mean([h['success'] for h in last_window])
                    m_avg = 0
                    for h in history:
                        m_avg = 0.9 * m_avg + 0.1 * h['success']
                        if m_avg >= 0.8:
                            conv_step = h['step']; break
            
            results.append({
                'Trial': trial_num,
                'Score': row['value'],
                'FinalSucc': "{0:.1%}".format(final_succ),
                'ConvAt': conv_step,
                'Scale': row.get('params_reward_scale', 0.0),
                'Penalty': row.get('params_time_penalty', 0.0),
                'Entropy': row.get('params_target_entropy', 0.0)
            })

        if results:
            report_df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
            print(report_df.to_string(index=False))
        else:
            print("No completed trials yet.")
        print("-" * 95)

if __name__ == "__main__":
    inspect_tuning()
