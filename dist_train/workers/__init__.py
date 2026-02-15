# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch.distributed as dist
import numpy as np
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
    if rank == 0:
        from tqdm import tqdm
        total_cycles = int(settings.dur) * config['cycles_per_epoch']
        pbar = tqdm(range(total_cycles), desc="SPECTRA RL (Rank 00)")
    else:
        pbar = range(int(settings.dur) * config['cycles_per_epoch'])

    last_succ = 0.0
    last_ret = 0.0
    last_alpha = 1.0

    for i in pbar:
        # One cycle consists of rollouts and (if buffer is ready) updates
        manager.do_cycle()
        
        # Periodic epoch-level logic (evaluation, checkpointing)
        if (i + 1) % config['cycles_per_epoch'] == 0:
            manager.curr_epoch += 1
            
            ready = manager.group_is_ready()
            # Perform evaluation
            stats, episodes = manager.eval_wrapper()
            manager.log_eval_results(stats, episodes)
            
            # Update metrics for tqdm
            if len(stats) > 0:
                mean_stats = np.mean(stats, axis=0)
                last_succ = mean_stats[0]
                last_ret = mean_stats[2] # Adjust index based on summary keys
                if len(mean_stats) > 12:
                    last_alpha = mean_stats[12]
            
            if rank == 0:
                manager.checkpoint()
                manager.replay_buffer.profile(0.001)
            
            dist.barrier()
            
            # Learning rate decay
            for gp in manager.optim.param_groups:
                gp['lr'] *= config.get("epoch_lr_decay", 1.0)

        # Update tqdm status every cycle
        if rank == 0:
            status = "Training" if manager.group_is_ready() else "Filling Buffer"
            buf_size = int(manager.replay_buffer.size)
            steps = int(manager.agent_model.train_steps.item())
            pbar.set_postfix({
                'Stat': status,
                'Succ': "{:.2f}".format(last_succ),
                'Ret': "{:.1f}".format(last_ret),
                'Alpha': "{:.4f}".format(last_alpha),
                'Step': steps
            })
