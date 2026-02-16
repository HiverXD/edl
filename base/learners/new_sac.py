# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from agents.maze_agents.modules.value_function import Critic
from agents.maze_agents.modules.policy import ReparamTrickPolicy

class NewSAC(nn.Module):
    """
    Standalone SAC-v2 implementation.
    Decoupled from the legacy BaseLearner/DistTrain hierarchy for clean debugging.
    """
    def __init__(self, env, hidden_size=256, learning_rate=3e-4, gamma=0.99, tau=0.005, 
                 auto_alpha=True, target_entropy=None, skill_dim=10):
        super().__init__()
        
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        self.device = torch.device("cpu") # Force CPU for stability
        
        # 1. Networks
        # ReparamTrickPolicy handles Gaussian sampling + Tanh squashing
        # We assume goal_size = skill_dim (embedding dimension)
        self.actor = ReparamTrickPolicy(env, hidden_size=hidden_size, goal_size=skill_dim, normalize_inputs=False)
        
        # Twin Critics
        self.q1 = Critic(env, hidden_size=hidden_size, goal_size=skill_dim, normalize_inputs=False)
        self.q2 = Critic(env, hidden_size=hidden_size, goal_size=skill_dim, normalize_inputs=False)
        self.q1_target = Critic(env, hidden_size=hidden_size, goal_size=skill_dim, normalize_inputs=False)
        self.q2_target = Critic(env, hidden_size=hidden_size, goal_size=skill_dim, normalize_inputs=False)
        
        # Sync weights
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Freeze target networks
        for p in self.q1_target.parameters(): p.requires_grad = False
        for p in self.q2_target.parameters(): p.requires_grad = False
        
        # 2. Optimizers (Explicitly Separated)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=learning_rate)
        
        # 3. Alpha Tuning
        if self.auto_alpha:
            if target_entropy is None:
                # heuristic: -dim(A)
                self.target_entropy = -float(env.action_size)
            else:
                self.target_entropy = float(target_entropy)
                
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.alpha_val = 0.2

        self.to(self.device)

    @property
    def alpha(self):
        return self.log_alpha.exp() if self.auto_alpha else torch.tensor(self.alpha_val).to(self.device)

    def select_action(self, state, skill, deterministic=False):
        """
        state: (obs_dim,) numpy or torch tensor
        skill: (skill_dim,) torch tensor
        """
        with torch.no_grad():
            if torch.is_tensor(state):
                s = state.float().unsqueeze(0).to(self.device)
            else:
                s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            z = skill.float().unsqueeze(0).to(self.device)
            
            # Policy returns: action, action_logit (pre-tanh), log_prob, entropy
            # greedy=True -> Deterministic (Mean)
            action, _, _, _ = self.actor(s, z, greedy=deterministic)
            return action.squeeze(0).cpu().numpy()

    def update(self, batch):
        """
        Explicit update step.
        Batch is a dict of tensors: state, action, reward, next_state, done, skill
        """
        s = batch['state'].to(self.device)
        a = batch['action'].to(self.device)
        r = batch['reward'].to(self.device)
        ns = batch['next_state'].to(self.device)
        d = batch['done'].to(self.device)
        z = batch['skill'].to(self.device) # Skill embedding
        
        # --- 1. Alpha Update ---
        if self.auto_alpha:
            with torch.no_grad():
                # Policy returns: action, action_logit, lprobs, n_ent
                _, _, log_pi, _ = self.actor(s, z)
                # Sum log_pi across action dimensions
                log_pi_sum = log_pi.sum(dim=1)
            
            alpha_loss = -(self.log_alpha * (log_pi_sum + self.target_entropy).detach()).mean()
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            curr_alpha = self.alpha.detach()
        else:
            alpha_loss = 0.0
            curr_alpha = self.alpha

        # --- 2. Critic Update ---
        with torch.no_grad():
            # Sample next action
            next_action, _, next_log_pi, _ = self.actor(ns, z)
            next_log_pi_sum = next_log_pi.sum(dim=1)
            
            # Target Q
            q1_t = self.q1_target(ns, next_action, z)
            q2_t = self.q2_target(ns, next_action, z)
            min_q_t = torch.min(q1_t, q2_t) - curr_alpha * next_log_pi_sum
            
            # Bellman Target
            target_q = r + self.gamma * (1 - d) * min_q_t

        # Current Q
        q1_pred = self.q1(s, a, z)
        q2_pred = self.q2(s, a, z)
        
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)
        
        # Update Q1
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()
        
        # Update Q2
        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # --- 3. Actor Update ---
        # Resample action with gradients
        new_action, _, log_pi_new, _ = self.actor(s, z)
        log_pi_new_sum = log_pi_new.sum(dim=1)
        
        q1_pi = self.q1(s, new_action, z)
        q2_pi = self.q2(s, new_action, z)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        # Actor Loss: alpha * log_pi - Q
        actor_loss = (curr_alpha * log_pi_new_sum - min_q_pi).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # --- 4. Soft Update ---
        self._polyak_update(self.q1, self.q1_target)
        self._polyak_update(self.q2, self.q2_target)
        
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "p_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item() if self.auto_alpha else 0.0,
            "alpha": curr_alpha.item(),
            "avg_q": min_q_pi.mean().item()
        }

    def _polyak_update(self, source, target):
        for p, p_targ in zip(source.parameters(), target.parameters()):
            p_targ.data.copy_(p_targ.data * (1 - self.tau) + p.data * self.tau)

    def load_checkpoint(self, path):
        ckpt = torch.load(path)
        self.actor.load_state_dict(ckpt['actor'])
        self.q1.load_state_dict(ckpt['q1'])
        self.q2.load_state_dict(ckpt['q2'])
        if self.auto_alpha and 'log_alpha' in ckpt:
            self.log_alpha.data = ckpt['log_alpha']

    def save_checkpoint(self, path):
        state = {
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
        }
        if self.auto_alpha:
            state['log_alpha'] = self.log_alpha.data
        torch.save(state, path)
