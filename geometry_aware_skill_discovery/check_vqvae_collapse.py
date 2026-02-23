# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import torch
import numpy as np
import pickle
import yaml
from argparse import ArgumentParser
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.maze_agents.toy_maze.skill_discovery.edl import VQVAEDiscriminator

def analyze_collapse():
    parser = ArgumentParser()
    parser.add_argument("--maze_type", type=str, default="square_corridor")
    parser.add_argument("--base_dir", type=str, default="logs/edl/vqvae")
    args = parser.parse_args()

    seeds = [42, 123, 999]
    print("\n" + "="*60)
    print(" VQ-VAE CODEBOOK COLLAPSE ANALYSIS: {0}".format(args.maze_type.upper()))
    print("="*60)

    data_path = "data/oracle_transitions/curriculum/{0}/identity/vqvae_oracle.pkl".format(args.maze_type)
    if not os.path.exists(data_path):
        print("Data not found at {0}".format(data_path))
        return
        
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)['dataset']
    
    for seed in seeds:
        model_dir = os.path.join(args.base_dir, args.maze_type, "curriculum/identity", str(seed))
        model_path = os.path.join(model_dir, "model.pth.tar")
        
        if not os.path.exists(model_path):
            print("Model not found for seed {0}".format(seed))
            continue
        
        # Load Model with Normalization enabled to match training
        model = VQVAEDiscriminator(state_size=2, hidden_size=256, codebook_size=10, code_size=10, normalize_inputs=True)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        with torch.no_grad():
            z_e_x = model.encoder(dataset)
            distances = model.vq.compute_distances(z_e_x)
            selected_indices = torch.argmin(distances, dim=1).numpy()
            
        counts = np.bincount(selected_indices, minlength=10)
        active_skills = np.sum(counts > 0)
        
        probs = counts / counts.sum()
        perplexity = np.exp(-np.sum(probs * np.log(probs + 1e-10)))
        
        print("\n>>> Seed {0}".format(seed))
        print("  Active Skills:  {0} / 10".format(active_skills))
        print("  Usage Dist:     {0}".format(counts))
        print("  Codebook Perp:  {0:.2f}".format(perplexity))

    print("\n" + "="*60)

if __name__ == "__main__":
    analyze_collapse()
