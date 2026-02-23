# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import json
import yaml
import torch
import argparse
import numpy as np
import pickle
from tqdm import tqdm
from agents.maze_agents.toy_maze.skill_discovery.edl import VQVAEDiscriminator

def load_gasd_dataset(config):
    exp_cfg = config['experiment']
    vq_cfg = config['vqvae']
    data_path = os.path.join("data/oracle_transitions", exp_cfg['exp_name'], exp_cfg['maze_type'], vq_cfg['mode'], "vqvae_oracle.pkl")
    with open(data_path, 'rb') as f: data = pickle.load(f)
    return data['dataset']

def train():
    parser = argparse.ArgumentParser("Train Advanced VQ-VAE (EMA).")
    parser.add_argument('--config-path', type=str, default="edl_config.yaml")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=0)
    args = parser.parse_args()

    # 1. Setup
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    with open(args.config_path, 'r') as f: config = yaml.load(f)
    
    maze_type = config['experiment']['maze_type']
    exp_name = config['experiment']['exp_name']
    vq_cfg = config['vqvae']
    
    # Force Advanced Mode by ensuring decay is passed
    vae_args = {
        'hidden_size': vq_cfg['hidden_size'],
        'codebook_size': vq_cfg['codebook_size'],
        'code_size': vq_cfg['code_size'],
        'beta': 0.25,
        'decay': vq_cfg.get('decay', 0.99),
        'epsilon': vq_cfg.get('epsilon', 1e-5),
        'normalize_inputs': vq_cfg.get('normalize_inputs', True)
    }
    
    dataset = load_gasd_dataset(config)
    exp_dir = os.path.join("logs/edl/advanced_vqvae", maze_type, exp_name, str(args.seed))
    if not os.path.exists(exp_dir): os.makedirs(exp_dir)
    
    # 2. Model & Optim
    model = VQVAEDiscriminator(state_size=2, **vae_args)
    if vae_args['normalize_inputs']: model.update_normalizer(dataset=dataset)
    
    # Filter parameters: EMA weights (requires_grad=False) should not be in optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=vq_cfg['learning_rate'])
    
    # 3. Loop
    indices = list(range(dataset.size(0)))
    total_iters = args.epochs if args.epochs > 0 else vq_cfg['epochs']
    pbar = tqdm(range(total_iters), desc="Advanced VQ-VAE (EMA) Seed {0}".format(args.seed))
    
    for _ in pbar:
        batch_indices = np.random.choice(indices, size=vq_cfg['batch_size'])
        batch = { 'next_state': dataset[batch_indices] }
        
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({'Loss': "{:.4f}".format(loss.item())})

    # 4. Save
    torch.save(model.state_dict(), os.path.join(exp_dir, "model.pth.tar"))
    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump({'vae_args': vae_args, 'maze_type': maze_type}, f, indent=4)
    print("\n[Success] Advanced VQ-VAE saved to {0}".format(exp_dir))

if __name__ == "__main__":
    train()
