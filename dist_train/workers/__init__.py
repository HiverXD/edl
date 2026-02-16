# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch.distributed as dist
import numpy as np
import os
import json
from dist_train.workers import baseline

episodic_off_policy_manager_lookup = {
    'baseline': baseline.EpisodicOffPolicy,
    'hierarchical': baseline.HierarchicalEpisodicOffPolicy
}

off_policy_manager_lookup = {
    'baseline': baseline.OffPolicy,
}

on_policy_manager_lookup = {
    'baseline': baseline.OnPolicy,
}

ppo_manager_lookup = {
    'baseline': baseline.PPO,
    'hierarchical': baseline.HierarchicalPPO
}

# For listing the current algorithms (see agents/base/algorithm_deecorators/) that belong to each manager group
on_policy_algos = []  # (ignore PPO here; it is unique)
off_policy_algos = ['sac', 'sac_v2']
episodic_off_policy_algos = ['ddpg', 'dqn']

def synchronous_worker(rank, config, settings):
    """Create a worker to play episodes on a given port and send the results to the trainer"""

    # Create a distributed process so the workers can share gradients and other such things
    dist.init_process_group(
        backend='gloo',
        init_method='tcp://127.0.0.1:43220',
        rank=rank,
        world_size=settings.N
    )
    print('Rank {:02d} worker successfully initiated the distributed process group!'.format(rank), flush=True)

    train_type = config['train_type']

    style = 'hierarchical' if 'hierarchical' in config['learner_type'].lower() else 'baseline'

    # Training is managed according to the PPO set up
    if train_type == 'ppo':
        manager_class = ppo_manager_lookup[style]

    # Doing some on-policy learning algorithm
    elif train_type in on_policy_algos:
        manager_class = on_policy_manager_lookup[style]

    # Doing some off-policy learning algorithm
    elif train_type in off_policy_algos:
        manager_class = off_policy_manager_lookup[style]

    # Doing some (episodic) off-policy learning algorithm
    elif train_type in episodic_off_policy_algos:
        manager_class = episodic_off_policy_manager_lookup[style]

    else:
        raise ValueError('Could not associate train_type "{}" with any known training manager'.format(train_type))

    # Create a manager object for this worker
    manager = manager_class(rank, config, settings)

    # Run through however many epochs we're supposed to
    # Use tqdm only for rank 0
    num_epochs = int(settings.dur)
    if rank == 0:
        from tqdm import tqdm
        # Progress bar now tracks epochs
        pbar = tqdm(range(num_epochs), desc="SPECTRA RL (Rank 00)", ncols=100, leave=True)
    else:
        pbar = range(num_epochs)

    last_succ = 0.0
    last_ret = 0.0
    last_alpha = 1.0

    for epoch_idx in pbar:
        # Each epoch consists of multiple cycles (Default to 1 for original EDL)
        num_cycles = config.get('cycles_per_epoch', 1)
        for cycle_idx in range(num_cycles):
            manager.do_cycle()
            
            # Optional: Update tqdm postfix inside cycles to show internal progress
            if rank == 0:
                status = "Training" if manager.group_is_ready() else "Filling Buffer"
                pbar.set_postfix({
                    'Cyc': "{}/{}".format(cycle_idx + 1, num_cycles),
                    'Stat': status,
                    'Succ': "{:.2f}".format(last_succ),
                    'Step': int(manager.agent_model.train_steps.item())
                })

        # Epoch-level logic (evaluation, checkpointing)
        manager.curr_epoch += 1
        
        # Perform evaluation
        stats, episodes = manager.eval_wrapper()
        manager.log_eval_results(stats, episodes)
        
        # Update metrics for tqdm
        if len(stats) > 0:
            mean_stats = np.mean(stats, axis=0)
            last_succ = mean_stats[0]
            last_ret = mean_stats[2]
            if len(mean_stats) > 12:
                last_alpha = mean_stats[12]
            
            # Log to training_stats.json
            if rank == 0:
                stats_file = os.path.join(manager.exp_dir, "training_stats.json")
                history = []
                if os.path.exists(stats_file):
                    with open(stats_file, 'r') as f:
                        try: history = json.load(f)
                        except: pass
                
                # Map all metrics by name for absolute clarity
                epoch_data = {'epoch': manager.curr_epoch}
                for idx, key in enumerate(manager.agent_model.ep_summary_keys):
                    if idx < len(mean_stats):
                        epoch_data[key] = float(mean_stats[idx])
                
                # Redundant keys for tqdm compatibility
                epoch_data['success_rate'] = epoch_data.get('success', 0.0)
                epoch_data['avg_return'] = epoch_data.get('cum_rew_total', 0.0)
                
                history.append(epoch_data)
                with open(stats_file, 'w') as f:
                    json.dump(history, f, indent=4)
        
        if rank == 0:
            manager.checkpoint()
            manager.replay_buffer.profile(0.001)
        
        dist.barrier()
        
        # Learning rate decay
        for gp in manager.optim.param_groups:
            gp['lr'] *= config.get("epoch_lr_decay", 1.0)
