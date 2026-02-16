# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import json
import torch
import torch.nn as nn
from .base import StochasticAgent
from agents.maze_agents.toy_maze.env import Env
from base.modules.normalization import DatasetNormalizer
from agents.maze_agents.modules.density import VQVAEDensity
from agents.maze_agents.modules import StochasticPolicy, Value
from base.learners.skill_discovery.edl import BaseEDLLearner, BaseEDLSiblingRivalryLearner


class DistanceStochasticAgent(StochasticAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_keys += ['goal']  # 'goal' is only used for visualization purposes

    def _make_modules(self, policy, skill_embedding, vae):
        super()._make_modules(policy, skill_embedding)
        self.vae = vae

    def step(self, do_eval=False):
        super().step(do_eval=do_eval)
        self.episode[-1]['goal'] = self.env.goal.detach()

    def reset(self, skill=None, *args, **kwargs):
        self.reset_skill(skill)
        # If the skill is a continuous vector (interpolated), pass it directly to the decoder.
        # Otherwise, treat it as a discrete skill index.
        if self.curr_skill.dim() > 0 and self.curr_skill.numel() > 1:
            # It's an interpolated vector, feed it directly to the decoder
            goal_state = self.vae.decoder(self.curr_skill.unsqueeze(0)).squeeze(0).detach().numpy()
        else:
            # It's a scalar index, use the existing logic
            goal_state = self.vae.get_centroids(dict(skill=self.curr_skill.view([]))).detach().numpy()
        kwargs['goal'] = goal_state
        self.env.reset(*args, **kwargs)
        self.episode = []

    def preprocess_skill(self, curr_skill):
        assert curr_skill is not None
        return self.skill_embedding(curr_skill).detach()


class SiblingRivalryStochasticAgent(DistanceStochasticAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_keys += ['antigoal']


class VQVAEDiscriminator(VQVAEDensity):
    def __init__(self, state_size, hidden_size, codebook_size, code_size, beta=0.25, **kwargs):
        super().__init__(num_skills=0, state_size=state_size, hidden_size=hidden_size, codebook_size=codebook_size,
                         code_size=code_size, beta=beta, **kwargs)
        self.softmax = nn.Softmax(dim=1)

    def _make_normalizer_module(self):
        self.normalizer = DatasetNormalizer(self.input_size) if self.normalize_inputs else None

    def compute_logprob(self, batch, with_codes=False):
        x = batch[self.input_key]
        z_e_x = self.encoder(x)
        z_q_x, selected_codes = self.vq.straight_through(z_e_x)
        x_ = self.decoder(z_q_x)
        if self.normalizes_inputs:
            x_ = self.normalizer.denormalize(x_)
        logprob = -1. * self.mse_loss(x, x_).sum(dim=1)
        if with_codes:
            return logprob, z_e_x, selected_codes
        else:
            return logprob

    def compute_logprob_under_latent(self, batch, z=None):
        x = batch[self.input_key]
        if z is None:
            z = batch['skill']
        z_q_x = self.vq.embedding(z).detach()
        x_ = self.decoder(z_q_x).detach()
        if self.normalizes_inputs:
            x_ = self.normalizer.denormalize(x_)
        logprob = -1. * self.mse_loss(x, x_).sum(dim=1)
        return logprob

    def log_approx_posterior(self, batch):
        x, z = batch[self.input_key], batch['skill']
        z_e_x = self.encoder(x)
        codebook_distances = self.vq.compute_distances(z_e_x)
        p = self.softmax(codebook_distances)
        p_z = p[torch.arange(0, p.shape[0]), z]
        return torch.log(p_z)

    def surprisal(self, batch):
        with torch.no_grad():
            return self.compute_logprob_under_latent(batch).detach()


class EDLLearner(BaseEDLLearner):

    def __init__(self, vae_logdir, **kwargs):
        self._parse_init_args(vae_logdir, **kwargs)
        
        # Trigger IM logic in BaseLearner
        if 'im_params' not in kwargs:
            kwargs['im_params'] = {'nu': 1.0}
            
        # Clean up remaining training-specific keys
        for k in ['dur', 'num_workers', 'log_dir', 'learning_rate', 'batch_size', 'horizon', 'mini_batch_size']:
            kwargs.pop(k, None)
            
        super().__init__(**kwargs)
        
        # Restore im reference to the vae
        self.im = self.vae

    def _parse_init_args(self, vae_logdir, **kwargs):
        vae_logdir = str(vae_logdir)
        if not os.path.isabs(vae_logdir):
            root_dir = os.environ.get("ROOT_DIR", os.getcwd())  # useful when loading experiments from a notebook
            vae_logdir = os.path.join(root_dir, vae_logdir)
        assert os.path.exists(vae_logdir), "Directory not found: {}".format(vae_logdir)
        self.vae_args = json.load(open(os.path.join(vae_logdir, "config.json")))["vae_args"]
        self.vae_checkpoint_path = os.path.join(vae_logdir, "model.pth.tar")

    def create_env(self):
        return Env(**self.env_params)

    def _make_im_modules(self):
        """Satisfy BaseLearner. Returns the VAE which acts as IM."""
        return self.vae

    def _make_agent_modules(self):
        self.vae = VQVAEDiscriminator(state_size=self._dummy_env.state_size, **self.vae_args)
        self.vae.load_checkpoint(self.vae_checkpoint_path)
        kwargs = dict(env=self._dummy_env, hidden_size=self.hidden_size, num_layers=self.num_layers,
                      goal_size=self.vae.code_size, normalize_inputs=self.normalize_inputs)
        self.policy = StochasticPolicy(**kwargs)
        self.v_module = Value(use_antigoal=False, **kwargs)
        self.v_target = Value(use_antigoal=False, **kwargs)
        self.v_target.load_state_dict(self.v_module.state_dict())
        
        from agents.maze_agents.modules.value_function import Critic
        self.q1 = Critic(**kwargs)
        self.q2 = Critic(**kwargs)

    def get_curr_qs(self, batch, new_actions=None, q_i=1):
        action = new_actions if new_actions is not None else batch['action']
        q_module = self.q1 if q_i == 1 else self.q2
        return q_module(batch['state'], action, self.preprocess_skill(batch['skill']))

    def get_action_qs(self, batch, q_i=1):
        return self.get_curr_qs(batch, q_i=q_i)

    def get_policy_loss_and_actions(self, batch):
        """Standard Actor loss for SAC."""
        action, _, _, _ = self.policy(batch['state'], self.preprocess_skill(batch['skill']))
        # Maximize Q1(s, pi(s))
        q_val = self.q1(batch['state'], action, self.preprocess_skill(batch['skill']))
        p_loss = -q_val.mean()
        return p_loss, action

    def get_next_vs(self, batch):
        return self.v_target(batch['next_state'], self.preprocess_skill(batch['skill']))

    def soft_update(self):
        for target_param, param in zip(self.v_target.parameters(), self.v_module.parameters()):
            target_param.data.copy_(target_param.data * self.polyak + param.data * (1.0 - self.polyak))

    def _make_agent(self):
        return DistanceStochasticAgent(env=self.create_env(), policy=self.policy, skill_n=self.vae.codebook_size,
                                       skill_embedding=self.vae.vq.embedding, vae=self.vae)

    def get_values(self, batch):
        return self.v_module(
            batch['state'],
            self.preprocess_skill(batch['skill'])
        )

    def get_terminal_values(self, batch):
        return self.v_module(
            batch['next_state'][-1:],
            self.preprocess_skill(batch['skill'][-1:]),
        )

    def get_policy_lprobs_and_nents(self, batch):
        log_prob, n_ent, _ = self.policy(
            batch['state'],
            self.preprocess_skill(batch['skill']),
            action_logit=batch['action_logit']
        )
        return log_prob.sum(dim=1), n_ent

    def sample_policy_actions_and_lprobs(self, batch):
        """Sample new actions for SAC V-network updates."""
        action, action_logit, lprobs, n_ent = self.policy(
            batch['state'],
            self.preprocess_skill(batch['skill'])
        )
        return action, lprobs.sum(dim=1)


class EDLSiblingRivalryLearner(BaseEDLSiblingRivalryLearner, EDLLearner):
    def __init__(self, **kwargs):
        self._parse_init_args(**kwargs)
        super().__init__(**kwargs)

    def _make_agent_modules(self):
        self.vae = VQVAEDiscriminator(state_size=self._dummy_env.state_size, **self.vae_args)
        self.vae.load_checkpoint(self.vae_checkpoint_path)
        kwargs = dict(env=self._dummy_env, hidden_size=self.hidden_size, num_layers=self.num_layers,
                      goal_size=self.vae.code_size, normalize_inputs=self.normalize_inputs)
        self.policy = StochasticPolicy(**kwargs)
        self.v_module = Value(use_antigoal=self.use_antigoal, **kwargs)

    def _make_agent(self):
        return SiblingRivalryStochasticAgent(env=self.create_env(), policy=self.policy, skill_n=self.vae.codebook_size,
                                             skill_embedding=self.vae.vq.embedding, vae=self.vae)

    def get_values(self, batch):
        return self.v_module(
            batch['state'],
            self.preprocess_skill(batch['skill']),
            batch.get('antigoal', None)
        )

    def get_terminal_values(self, batch):
        if 'antigoal' in batch:
            antigoal = batch['antigoal'][-1:]
        else:
            antigoal = None
        return self.v_module(
            batch['next_state'][-1:],
            self.preprocess_skill(batch['skill'][-1:]),
            antigoal
        )

