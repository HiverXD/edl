# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import torch
import numpy as np
import yaml
import json
import random
import time
from collections import deque
from argparse import ArgumentParser
from tqdm import tqdm

from base.learners.sac_v2 import SACV2Learner
from geometry_aware_skill_discovery.reward import SPECTRAProvider
from agents.maze_agents.toy_maze.env.maze_env import Env

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def add(self, state, action, reward, next_state, done, skill):
        self.buffer.append((state, action, reward, next_state, done, skill))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, skill = zip(*batch)
        def _stack(x):
            if torch.is_tensor(x[0]): return torch.stack(x).float().cpu()
            return torch.tensor(x).float().cpu()
        return {
            'state': _stack(state), 'action': _stack(action), 'reward': _stack(reward),
            'next_state': _stack(next_state), 'done': _stack(done), 'skill': torch.stack(skill).float().cpu() 
        }
    def __len__(self): return len(self.buffer)

def train():
    parser = ArgumentParser(description="Official SPECTRA RL Training Script.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--reward_type", type=str, default="static")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Setup
    with open(args.config, 'r') as f: config = yaml.load(f)
    common = config['rl']['common']
    maze_type = config['experiment']['maze_type']
    exp_name = config['experiment']['exp_name']
    
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    
    # 2. Components
    env = Env(n=50, maze_type=maze_type, done_on_success=True)
    provider_kwargs = {
        'sparse_bonus': common.get('sparse_bonus', 0.0),
        'success_threshold': common.get('success_threshold', 0.15),
        'gaussian_bonus': common.get('gaussian_bonus', 0.0),
        'gaussian_std': common.get('gaussian_std', 0.5),
        'reward_scale': config['rl'][args.reward_type].get('reward_scale', 1.0),
        'time_penalty': config['rl'][args.reward_type].get('time_penalty', 0.0)
    }
    provider = SPECTRAProvider(maze_type=maze_type, exp_name=exp_name, **provider_kwargs)
    
    agent = SACV2Learner(env=env, hidden_size=256, learning_rate=3e-4, gamma=0.99,
                         target_entropy=common.get('target_entropy', -2.0),
                         skill_dim=provider.centroids_psi.shape[1],
                         skill_n=provider.n_skills)
    
    # Load actual centroids into the agent's embedding layer
    agent.skill_embedding.weight.data.copy_(provider.centroids_psi)
    
    buffer = ReplayBuffer(capacity=1000000)
    
    # 3. Path Standardization
    log_dir = os.path.join("logs/rl", maze_type, exp_name, args.reward_type, str(args.seed))
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    model_path = os.path.join(log_dir, "model.pth")
    stats_path = os.path.join(log_dir, "training_stats.json")
    
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    start_step = 1; history = []
    if os.path.exists(model_path):
        print("Resuming from checkpoint: {0}".format(model_path))
        agent.load_checkpoint(model_path)
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                try: 
                    history = json.load(f)
                    if history: start_step = history[-1]['step']
                except: pass

    # 4. Training Loop
    num_epochs = common.get('dur', 1000)
    total_steps = num_epochs * 1000 
    start_steps = 5000 if start_step == 1 else 0 
    
    # [FIX] Get reward gamma from config (crucial for PBRS)
    rew_gamma = config['rl'][args.reward_type].get('pbrs_gamma', 0.99)
    
    # Success tracking for tqdm
    success_window = deque(maxlen=20)
    
    state = env.reset()
    skill_idx = np.random.randint(0, provider.n_skills)
    current_goal_psi = provider.centroids_psi[skill_idx]
    env.reset(goal=provider.get_goal_for_skill(skill_idx))
    state = env.state
    
    ep_reward = 0; ep_steps = 0; ep_breakdowns = []
    loss_stats = {'q1_loss': 0.0, 'p_loss': 0.0, 'alpha': 1.0}
    epoch_data = {'q_loss': 0.0, 'alpha': 1.0} 
    success_streak = 0 # Local variable for early stopping
    
    # 5. Training Loop
    pbar = tqdm(total=total_steps, desc="Learning", ncols=90)
    if start_step > 1: pbar.update(start_step)
    
    try:
        for step in range(start_step, total_steps + 1):
            pbar.update(1)
            if step < start_steps:
                action = np.random.uniform(-env.action_range, env.action_range, size=(2,))
            else:
                action = agent.select_action(state, current_goal_psi)
            
            env.step(action); next_state = env.state; done = env.is_done; success = env.is_success
            
            def _to_t(x): return x if torch.is_tensor(x) else torch.from_numpy(x).float()
            s_t, ns_t = _to_t(state).unsqueeze(0), _to_t(next_state).unsqueeze(0)
            reward = provider.compute_reward(s_t, ns_t, torch.tensor([skill_idx]), gamma=rew_gamma, reward_type=args.reward_type).item()
            ep_breakdowns.append(provider._last_breakdown.copy())
            
            buffer.add(state, action, reward, next_state, float(done), current_goal_psi)
            state = next_state; ep_reward += reward; ep_steps += 1
            
            # E. Update (Standard Off-policy: 1 update per 1 step)
            if step >= start_steps and len(buffer) >= 256:
                # To maintain the same 'training intensity' as the epoch-based version,
                # we perform 1 update every step.
                loss_stats = agent.update(buffer.sample(256))
            
            if step % 10000 == 0:
                agent.save_checkpoint(model_path)

            if done or ep_steps >= 50:
                success_window.append(int(success))
                avg_succ_rate = np.mean(success_window) if success_window else 0.0
                
                # --- [NEW] Early Stopping Logic ---
                # 1. Success-based Early Exit: 1.0 success maintained
                if avg_succ_rate >= 1.0:
                    success_streak += ep_steps
                    if success_streak >= 500: # Maintain 1.0 for 500 steps
                        pbar.write(">>> Goal maintained! Early termination at step {0}".format(step))
                        raise StopIteration # Use custom exception to trigger final save
                else:
                    success_streak = 0
                
                # 2. Failure-based Pruning: Below 10% at 40% of time
                if step > (total_steps * 0.4) and avg_succ_rate < 0.10:
                    pbar.write(">>> Pruning: Low success ({0:.2f}) at 40% mark.".format(avg_succ_rate))
                    raise StopIteration
                # ---------------------------------

                dist_to_g = env.dist(env.state, env.goal).item()
                avg_breakdown = {k: np.mean([b[k] for b in ep_breakdowns]) for k in ep_breakdowns[0].keys()} if ep_breakdowns else {}
                
                # Log to history
                epoch_data = {
                    'step': step, 'epoch': len(history) + 1, 'success': int(success), 'dist_to_goal': dist_to_g,
                    'avg_return': ep_reward, 'alpha': loss_stats.get('alpha', 1.0),
                    'q_loss': loss_stats.get('q1_loss', 0.0), 'p_loss': loss_stats.get('p_loss', 0.0),
                    'rew_pot': avg_breakdown.get('potential', 0.0), 'rew_pen': avg_breakdown.get('penalty', 0.0),
                    'rew_gauss': avg_breakdown.get('gauss', 0.0), 'rew_bon': avg_breakdown.get('bonus', 0.0)
                }
                history.append(epoch_data)
                
                # Periodically save JSON to disk (every 100 episodes)
                if len(history) % 100 == 0:
                    with open(stats_path, 'w') as f: json.dump(history, f, indent=4)
                
                # Reset Episode
                skill_idx = np.random.randint(0, provider.n_skills)
                current_goal_psi = provider.centroids_psi[skill_idx]
                env.reset(goal=provider.get_goal_for_skill(skill_idx))
                state = env.state; ep_reward = 0; ep_steps = 0; ep_breakdowns = []
    
            if step % 100 == 0:
                # Update Tqdm Dashboard
                avg_succ_rate = np.mean(success_window) if success_window else 0.0
                pbar.set_postfix({
                    'Succ': "{:.2f}".format(avg_succ_rate),
                    'Rew': "{:.1f}".format(ep_reward),
                    'Q-L': "{:.3f}".format(loss_stats.get('q1_loss', 0.0)),
                    'Alpha': "{:.3f}".format(loss_stats.get('alpha', 1.0))
                })
    
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    except StopIteration:
        print("\nTraining terminated early.")
    finally:
        with open(stats_path, 'w') as f: json.dump(history, f, indent=4)
        agent.save_checkpoint(model_path)
        print("Progress saved.")

if __name__ == "__main__":
    train()
