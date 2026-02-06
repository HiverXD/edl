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
import collections
import json

# Add project root to path
sys.path.append(os.getcwd())

# Define Step namedtuple for pickle compatibility
Step = collections.namedtuple('Step', 'agent_state, action, episode_done')

class TransitionDataset(Dataset):
    def __init__(self, data_path, mean=None, std=None):
        print("Loading oracle data from {0}...".format(data_path))
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.transitions = data['raw_transitions']
        self.states = np.array([t[0] for t in self.transitions], dtype=np.float32)
        self.next_states = np.array([t[2] for t in self.transitions], dtype=np.float32)
        
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
            print("Using provided normalization stats.")
        else:
            self.mean = self.states.mean(axis=0)
            self.std = self.states.std(axis=0) + 1e-6
            print("Calculated new normalization stats.")
            
        print("Dataset loaded. Total transitions: {0}".format(len(self.transitions)))
        
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        s = (self.states[idx] - self.mean) / self.std
        s_next = (self.next_states[idx] - self.mean) / self.std
        return torch.from_numpy(s), torch.from_numpy(s_next)

class LaplacianEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        super().__init__()
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

class ALLOTrainer:
    def __init__(self, model, device, output_dim, lr=1e-3, dual_lr=1e-3, rho_init=1.0, rho_lr=1e-2):
        self.model = model.to(device)
        self.device = device
        self.output_dim = output_dim
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.beta = torch.zeros(output_dim, output_dim, device=device)
        self.rho = rho_init
        self.rho_lr = rho_lr
        self.dual_lr = dual_lr
        # self.weights = torch.ones(output_dim, dtype=torch.float32, device=device)
        self.weights = torch.arange(output_dim, 0, -1, dtype=torch.float32, device=device)

    def train_step(self, s, s_next):
        self.model.train()
        phi_s = self.model(s)
        phi_next = self.model(s_next)
        
        batch_size = s.size(0)
        phi_mean = phi_s.mean(dim=0)
        mean_penalty = phi_mean.pow(2).sum()
        
        phi_centered = phi_s - phi_s.mean(dim=0, keepdim=True)
        gram = torch.mm(phi_centered.t(), phi_centered) / (batch_size - 1)
        error_matrix = gram - torch.eye(self.output_dim, device=self.device)
        
        per_dim_graph_loss = (phi_s - phi_next).pow(2).mean(dim=0)
        graph_loss = (per_dim_graph_loss * self.weights).sum()
        
        dual_loss = (self.beta * error_matrix).sum()
        barrier_loss = (self.rho / 2.0) * error_matrix.pow(2).sum()
        
        total_loss = graph_loss + dual_loss + barrier_loss + 10.0 * mean_penalty
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            self.beta += self.dual_lr * error_matrix
            self.rho += self.rho_lr * error_matrix.pow(2).sum().item()
            self.rho = min(self.rho, 1000.0) 
            
        return {
            'loss': total_loss.item(),
            'graph_loss': per_dim_graph_loss.sum().item(),
            'eigenvalues': per_dim_graph_loss.detach().cpu().numpy().tolist(),
            'ortho_error': error_matrix.pow(2).sum().item(),
            'max_error': error_matrix.abs().max().item(),
            'rho': self.rho
        }

def save_checkpoint(model, trainer, dataset, log_history, args, model_file, stats_file):
    torch.save({
        'state_dict': model.state_dict(),
        'eigenvalues': log_history[-1]['eigenvalues'] if log_history else [],
        'duals': trainer.beta.cpu(),
        'rho': trainer.rho,
        'mean': dataset.mean, 'std': dataset.std, 'args': args
    }, model_file)
    
    with open(stats_file, "w") as f:
        json.dump(log_history, f, indent=2)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze_type", type=str, default="square_a")
    parser.add_argument("--data_path", type=str, default=None, help="Path to oracle transitions.")
    parser.add_argument("--save_dir", type=str, default="logs/laplacian_encoder")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dual_lr", type=float, default=1e-3)
    parser.add_argument("--rho_lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint")
    parser.add_argument("--additional_epochs", type=int, default=100, help="Epochs to add when resuming")
    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = "data/oracle_transitions/default/{0}/transitions.pkl".format(args.maze_type)

    device = torch.device("cpu")
    save_path = os.path.join(args.save_dir, args.maze_type, args.exp_name)
    model_file = os.path.join(save_path, "model.pth.tar")
    stats_file = os.path.join(save_path, "training_stats.json")

    checkpoint = {}
    log_history = []
    
    if args.resume:
        if os.path.exists(model_file):
            print("Resuming training from {0}".format(model_file))
            checkpoint = torch.load(model_file, map_location='cpu')
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    log_history = json.load(f)
                start_epoch = log_history[-1]['epoch']
            else:
                start_epoch = 0
            
            total_epochs = start_epoch + args.additional_epochs
            # Keep original architecture params
            args.dim = checkpoint['args'].dim
            args.hidden_dim = checkpoint['args'].hidden_dim
        else:
            print("Checkpoint not found. Starting from scratch.")
            args.resume = False
            start_epoch = 0
            total_epochs = args.epochs
    else:
        start_epoch = 0
        total_epochs = args.epochs

    # 1. Dataset
    dataset = TransitionDataset(args.data_path, 
                                mean=checkpoint.get('mean'), 
                                std=checkpoint.get('std'))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 2. Model
    model = LaplacianEncoder(input_dim=2, hidden_dim=args.hidden_dim, output_dim=args.dim)
    if args.resume:
        model.load_state_dict(checkpoint['state_dict'])

    # 3. Trainer
    trainer = ALLOTrainer(
        model=model, device=device, output_dim=args.dim, 
        lr=args.lr, dual_lr=args.dual_lr, rho_lr=args.rho_lr
    )
    if args.resume:
        if 'duals' in checkpoint:
            trainer.beta.data.copy_(checkpoint['duals'])
        trainer.rho = checkpoint.get('rho', 1.0)

    if not os.path.exists(save_path): os.makedirs(save_path)

    print("\n--- Starting Refined ALLO Training ---")
    
    # Outer progress bar for epochs
    pbar = tqdm(range(start_epoch, total_epochs), desc="Training", ascii=True)
    
    best_gloss = float('inf')
    best_ortho = float('inf')
    patience = 10
    patience_counter = 0
    ortho_threshold = 1e-3

    for epoch in pbar:
        epoch_metrics = collections.defaultdict(list)
        for batch_s, batch_s_next in dataloader:
            m = trainer.train_step(batch_s, batch_s_next)
            for k, v in m.items(): epoch_metrics[k].append(v)
        
        # Average metrics for the epoch
        avg_metrics = {k: np.mean(v) if k != 'eigenvalues' else np.mean(v, axis=0).tolist() 
                       for k, v in epoch_metrics.items()}
        avg_metrics['epoch'] = epoch + 1
        log_history.append(avg_metrics)

        # Update tqdm status
        pbar.set_postfix({
            'GLoss': "{0:.4f}".format(avg_metrics['graph_loss']),
            'OrthoErr': "{0:.6f}".format(avg_metrics['ortho_error']),
            'Rho': "{0:.1f}".format(avg_metrics['rho'])
        })

        # Periodic save and check
        if (epoch + 1) % 10 == 0:
            # Check for negative eigenvalues
            neg_check = any(ev < 0 for ev in avg_metrics['eigenvalues'])
            if neg_check:
                pbar.write("Epoch {0}: Warning - Negative eigenvalue estimates detected.".format(epoch + 1))
            
            save_checkpoint(model, trainer, dataset, log_history, args, model_file, stats_file)


        # Early Stopping Logic: Stop if BOTH haven't improved significantly
        if avg_metrics['ortho_error'] < ortho_threshold:
            improved_gloss = avg_metrics['graph_loss'] < best_gloss * 0.999
            improved_ortho = avg_metrics['ortho_error'] < best_ortho * 0.999
            
            if not (improved_gloss or improved_ortho):
                patience_counter += 1
            else:
                patience_counter = 0
                if avg_metrics['graph_loss'] < best_gloss: best_gloss = avg_metrics['graph_loss']
                if avg_metrics['ortho_error'] < best_ortho: best_ortho = avg_metrics['ortho_error']
            
            if patience_counter >= patience:
                pbar.write("\nEarly stopping at epoch {0}: No significant improvement in structure or constraints.".format(epoch+1))
                break
        else:
            patience_counter = 0


    # Final save
    save_checkpoint(model, trainer, dataset, log_history, args, model_file, stats_file)
    print("\nTraining Finished. Model: {0}".format(model_file))

if __name__ == "__main__":
    train()