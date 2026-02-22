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
            
            # Derive reward type for path
            r_type = study_name.split(args.maze + "_")[-1]
            
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
