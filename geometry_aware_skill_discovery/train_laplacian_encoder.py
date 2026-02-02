# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import sys
import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

import collections
Step = collections.namedtuple('Step', 'agent_state, action, episode_done')

class TransitionDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # We use raw_transitions: list of (s, a, s_next)
        self.transitions = data['raw_transitions']
        self.states = np.array([t[0] for t in self.transitions], dtype=np.float32)
        self.next_states = np.array([t[2] for t in self.transitions], dtype=np.float32)
        
        # Compute mean and std for normalization
        self.mean = self.states.mean(axis=0)
        self.std = self.states.std(axis=0) + 1e-6
        
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        s = (self.states[idx] - self.mean) / self.std
        s_next = (self.next_states[idx] - self.mean) / self.std
        return torch.from_numpy(s), torch.from_numpy(s_next)

class LaplacianEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        super(LaplacianEncoder, self).__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze_type", type=str, default="square_a")
    parser.add_argument("--data_path", type=str, default="data/oracle_transitions/square_a/transitions.pkl")
    parser.add_argument("--save_dir", type=str, default="logs/laplacian_encoder")
    parser.add_argument("--dim", type=int, default=20, help="Dimension of Laplacian representation")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lambda_ortho", type=float, default=1.0, help="Weight for orthogonality loss")
    args = parser.parse_args()

    # Force CPU for compatibility with legacy PyTorch 1.2.0 on modern systems
    device = torch.device("cpu")
    print("Using device: {}".format(device))

    # 1. Load Data
    dataset = TransitionDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Create Model
    model = LaplacianEncoder(input_dim=2, hidden_dim=args.hidden_dim, output_dim=args.dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Weights for Graph Loss: Uniform weights (ones)
    # Using uniform weights allows the orthogonality loss to drive the separation of eigenfunctions
    # naturally, rather than forcing a specific order which might distort the geometry.
    weights = torch.ones(args.dim, dtype=torch.float32, device=device)

    # 3. Training Loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        total_graph_loss = 0
        total_ortho_loss = 0
        
        for batch_s, batch_s_next in dataloader:
            batch_s, batch_s_next = batch_s.to(device), batch_s_next.to(device)
            
            # Forward
            phi_s = model(batch_s)
            phi_s_next = model(batch_s_next)
            
            # Graph Loss: minimize weighted distance between connected states
            # (batch, dim)
            sq_diff = (phi_s - phi_s_next).pow(2)
            graph_loss = (sq_diff * weights).sum(dim=1).mean()
            
            # Orthogonality Loss: Cov(phi) should be Identity
            # We want E[phi * phi^T] = I
            # Compute Gram matrix within batch
            n = phi_s.size(0)
            # Center the features (important for covariance)
            phi_centered = phi_s - phi_s.mean(dim=0, keepdim=True)
            cov = (phi_centered.t() @ phi_centered) / (n - 1)
            
            # Penalty for off-diagonal (zero) and diagonal (one)
            ortho_loss = (cov - torch.eye(args.dim, device=device)).pow(2).mean()
            
            loss = graph_loss + args.lambda_ortho * ortho_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_graph_loss += graph_loss.item()
            total_ortho_loss += ortho_loss.item()
            
        if (epoch + 1) % 10 == 0:
            print("Epoch [{}/{}], Loss: {:.4f} (Graph: {:.4f}, Ortho: {:.4f})".format(
                epoch + 1, args.epochs, total_loss/len(dataloader), 
                total_graph_loss/len(dataloader), total_ortho_loss/len(dataloader)))

    # 4. Save Model
    save_path = os.path.join(args.save_dir, args.maze_type, "default")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    model_file = os.path.join(save_path, "model.pth.tar")
    torch.save({
        'state_dict': model.state_dict(),
        'mean': dataset.mean,
        'std': dataset.std,
        'args': args
    }, model_file)
    
    print("Model saved to {}".format(model_file))

if __name__ == "__main__":
    train()