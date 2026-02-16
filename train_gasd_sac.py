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

from base.learners.new_sac import NewSAC
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
    parser = ArgumentParser()
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
    
    agent = NewSAC(env=env, hidden_size=256, learning_rate=3e-4, gamma=0.99,
                   target_entropy=common.get('target_entropy', -2.0),
                   skill_dim=provider.centroids_psi.shape[1])
    buffer = ReplayBuffer(capacity=1000000)
    
    # 3. Resume Logic
    log_dir = os.path.join("logs/new_sac", maze_type, args.reward_type)
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    model_path = os.path.join(log_dir, "model.pth")
    stats_path = os.path.join(log_dir, "training_stats.json")
    
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
    
    state = env.reset()
    skill_idx = np.random.randint(0, provider.n_skills)
    current_goal_psi = provider.centroids_psi[skill_idx]
    env.reset(goal=provider.get_goal_for_skill(skill_idx))
    state = env.state
    
    ep_reward = 0; ep_steps = 0; ep_breakdowns = []
    print("\n--- Starting New SAC Training (JSON Logging) ---")
    
    try:
        for step in range(start_step, total_steps + 1):
            if step < start_steps:
                action = np.random.uniform(-env.action_range, env.action_range, size=(2,))
            else:
                action = agent.select_action(state, current_goal_psi)
            
            env.step(action); next_state = env.state; done = env.is_done; success = env.is_success
            
            def _to_t(x): return x if torch.is_tensor(x) else torch.from_numpy(x).float()
            s_t, ns_t = _to_t(state).unsqueeze(0), _to_t(next_state).unsqueeze(0)
            reward = provider.compute_reward(s_t, ns_t, torch.tensor([skill_idx]), gamma=0.99, reward_type=args.reward_type).item()
            ep_breakdowns.append(provider._last_breakdown.copy())
            
            buffer.add(state, action, reward, next_state, float(done), current_goal_psi)
            state = next_state; ep_reward += reward; ep_steps += 1
            
            loss_stats = {}
            if step >= start_steps and len(buffer) >= 256:
                batch = buffer.sample(256)
                loss_stats = agent.update(batch)
            
            if step % 10000 == 0:
                agent.save_checkpoint(model_path)
                print("\n--- Checkpoint Saved at Step {0} ---".format(step))

            if done or ep_steps >= 50:
                dist_to_g = env.dist(env.state, env.goal).item()
                avg_breakdown = {k: np.mean([b[k] for b in ep_breakdowns]) for k in ep_breakdowns[0].keys()} if ep_breakdowns else {}
                
                epoch_data = {
                    'step': step, 'epoch': len(history) + 1, 'success': int(success), 'dist_to_goal': dist_to_g,
                    'avg_return': ep_reward, 'alpha': loss_stats.get('alpha', 1.0),
                    'q_loss': loss_stats.get('q1_loss', 0.0), 'p_loss': loss_stats.get('p_loss', 0.0),
                    'rew_pot': avg_breakdown.get('potential', 0.0), 'rew_pen': avg_breakdown.get('penalty', 0.0),
                    'rew_gauss': avg_breakdown.get('gauss', 0.0), 'rew_bon': avg_breakdown.get('bonus', 0.0)
                }
                history.append(epoch_data)
                with open(stats_path, 'w') as f: json.dump(history, f, indent=4)

                if step % 50 == 0 or success:
                    print("Step {0:6d} | Rew: {1:6.2f} | Succ: {2} | Dist: {3:.2f} | Q-L: {4:.3f}".format(
                        step, ep_reward, int(success), dist_to_g, epoch_data['q_loss']))
                
                skill_idx = np.random.randint(0, provider.n_skills)
                current_goal_psi = provider.centroids_psi[skill_idx]
                env.reset(goal=provider.get_goal_for_skill(skill_idx))
                state = env.state; ep_reward = 0; ep_steps = 0; ep_breakdowns = []
                
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        agent.save_checkpoint(model_path)
        print("Final model saved.")

if __name__ == "__main__":
    train()
