# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import sys
import math
import json
import yaml
import torch
import shutil
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from agents.maze_agents.toy_maze.env.maze_env import Env
from agents.maze_agents.toy_maze.skill_discovery.edl import VQVAEDiscriminator


def sample_dataset(maze_type, num_samples, condition_fn=lambda x: True):
    """ condition_fn can be used to induce a prior over samples """
    env = Env(n=50, maze_type=maze_type, use_antigoal=False)
    dataset = np.zeros((num_samples, 2))
    for sample_idx in range(num_samples):
        done = False
        while not done:
            s = env.sample()
            done = condition_fn(s)
        dataset[sample_idx] = np.array(s)
    dataset = torch.from_numpy(dataset).float()
    return env, dataset

def load_gasd_dataset(config):
    """ Loads pre-processed GASD oracle data """
    exp_cfg = config['experiment']
    vq_cfg = config['vqvae']
    
    data_path = os.path.join("data/oracle_transitions", exp_cfg['exp_name'], exp_cfg['maze_type'], vq_cfg['mode'], "vqvae_oracle.pkl")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError("GASD data not found at {0}. Please run prepare_gasd_data.py first.".format(data_path))
        
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    dataset = data['dataset']
    print("Loaded GASD dataset from {0}. Shape: {1}".format(data_path, dataset.shape))
    return dataset

def open_experiment():
    parser = argparse.ArgumentParser("Train VQ-VAE.")
    parser.add_argument('--config-path', type=str, help='Path to experiment config file (.json or .yaml)')
    parser.add_argument('--log-dir', type=str, help='Parent directory that holds experiment log directories')
    parser.add_argument('--dur', type=int, default=0, help='Number of training iterations (overrides config if > 0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--mode', type=str, help='Override vqvae mode (identity or spectral)')
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config_path = args.config_path
    assert os.path.isfile(config_path)
    
    is_yaml = config_path.endswith('.yaml') or config_path.endswith('.yml')
    
    if is_yaml:
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        # Standardize YAML to match VQ-VAE training expectations
        maze_type = config['experiment']['maze_type']
        exp_name = config['experiment']['exp_name']
        mode = args.mode or config['vqvae']['mode']
        
        # Output directory: logs/edl/vqvae/{maze_type}/{exp_name}/{mode}/{seed}/
        log_root = args.log_dir or config['vqvae'].get('log_dir', "logs/edl/vqvae")
        exp_dir = os.path.join(log_root, maze_type, exp_name, mode, str(args.seed))
        
        # Load GASD dataset
        dataset = load_gasd_dataset(config)
        state_size = dataset.shape[1]
        
        # VAE Arguments from YAML
        vae_args = {
            'hidden_size': config['vqvae']['hidden_size'],
            'codebook_size': config['vqvae']['codebook_size'],
            'code_size': config['vqvae']['code_size'],
            'normalize_inputs': config['vqvae'].get('normalize_inputs', False) if mode == 'identity' else False
        }
        
        training_config = {
            'learning_rate': config['vqvae']['learning_rate'],
            'batch_size': config['vqvae']['batch_size'],
            'epochs': args.dur if args.dur > 0 else config['vqvae']['epochs'],
            'maze_type': maze_type
        }
        # For saving back to JSON
        full_config = {**training_config, 'vae_args': vae_args, 'sampler': 'gasd', 'mode': mode}
        
    else:
        # Legacy JSON support
        config = json.load(open(config_path))
        exp_name = config_path.split('/')[-1][:-5]
        exp_dir = os.path.join(args.log_dir, exp_name)
        
        if config['sampler'] == 'oracle':
            env, dataset = sample_dataset(config['maze_type'], config['num_samples'])
        elif config['sampler'] == 'smm':
            exp, dataset = load_smm_buffer(config['smm_exp_name'], config['smm_epoch'], notebook_mode=False)
            env = exp.learner.agent.env
            config['maze_type'] = env.maze_type
        else:
            raise ValueError("Invalid 'sampler' type")
            
        state_size = dataset.shape[1]
        full_config = config
        full_config['epochs'] = args.dur if args.dur > 0 else 50000

    print('Experiment directory is: {}'.format(exp_dir), flush=True)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
        
    # Save the config used for this run
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(full_config, f, indent=4)

    # Create VQ-VAE model
    model = VQVAEDiscriminator(state_size=state_size, **full_config['vae_args'])
    
    # Only update normalizer if explicitly enabled and not in Laplacian mode
    if full_config['vae_args'].get('normalize_inputs', False):
        print("Updating normalizer with dataset stats...")
        model.update_normalizer(dataset=dataset)
    else:
        print("Normalizer disabled (preserving input geometry).")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=full_config['learning_rate'])

    return model, optimizer, dataset, full_config, args, exp_dir


if __name__ == '__main__':
    # Interpret the arguments. Create the model, optimizer and dataset. Fetch the config file.
    model, optim, dataset, config, args, save_dir = open_experiment()

    # Training loop
    indices = list(range(dataset.size(0)))
    loss_list = []
    model.train()
    
    # Use epochs if specified (for GASD), otherwise use fixed iterations
    total_iters = config['epochs'] if config['sampler'] == 'gasd' else args.dur
    if total_iters == 0: total_iters = 50000 # Default fallback
    
    pbar = tqdm(range(total_iters), desc="VQ-VAE Training")
    for iter_idx in pbar:
        # Make batch
        batch_indices = np.random.choice(indices, size=config['batch_size'])
        batch = { 'next_state': dataset[batch_indices] }

        # Forward + backward pass
        optim.zero_grad()
        loss = model(batch)
        loss.backward()
        optim.step()

        # Log progress
        loss_list.append(loss.item())
        if iter_idx % 100 == 0:
            pbar.set_postfix({'Loss': "{0:.6f}".format(loss.item())})

    # Save model, config and losses
    model.eval()
    model_path = os.path.join(save_dir, "model.pth.tar")
    loss_path = os.path.join(save_dir, "loss.json")
    
    torch.save(model.state_dict(), model_path)
    with open(loss_path, 'wt') as f:
        json.dump(loss_list, f)
        
    print("\n[Done] VQ-VAE training complete. Model saved to {0}".format(model_path))