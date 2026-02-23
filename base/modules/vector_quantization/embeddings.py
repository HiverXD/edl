# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT


import torch
import torch.nn as nn
import torch.nn.functional as F
from .functions import vector_quantization, vector_quantization_st


class VQEmbedding(nn.Module):
    """ Standard VQ Module (Baseline) """
    def __init__(self, codebook_size, code_size, beta):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.code_size = int(code_size)
        self.beta = float(beta)
        self.embedding = nn.Embedding(self.codebook_size, self.code_size)
        self.embedding.weight.data.uniform_(-1./self.codebook_size, 1./self.codebook_size)
        self.mse_loss = nn.MSELoss(reduction='none')

    def quantize(self, z_e_x):
        return vector_quantization(z_e_x, self.embedding.weight)

    def straight_through(self, z_e_x):
        z_q_x, indices = vector_quantization_st(z_e_x, self.embedding.weight.detach())
        selected_codes = torch.index_select(self.embedding.weight, dim=0, index=indices)
        return z_q_x, selected_codes

    def forward(self, z_e_x, selected_codes=None):
        if selected_codes is None:
            _, selected_codes = self.straight_through(z_e_x)
        vq_loss = self.mse_loss(selected_codes, z_e_x.detach()).sum(dim=1)
        commitment_loss = self.mse_loss(z_e_x, selected_codes.detach()).sum(dim=1)
        loss = vq_loss + self.beta * commitment_loss
        return loss

    def compute_distances(self, inputs):
        with torch.no_grad():
            embedding_size = self.embedding.weight.size(1)
            inputs_flatten = inputs.view(-1, embedding_size)
            codebook_sqr = torch.sum(self.embedding.weight ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
            distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, self.embedding.weight.t(), alpha=-2.0, beta=1.0)
            return distances


class AdvancedVQEmbedding(VQEmbedding):
    """
    State-of-the-art VQ module:
    - EMA updates
    - Spherical VQ (L2 Normalization)
    - Random Restart for Dead Codes
    """
    def __init__(self, codebook_size, code_size, beta, decay=0.99, epsilon=1e-5, use_l2=True, restart_threshold=1.0):
        super().__init__(codebook_size, code_size, beta)
        self.decay = decay
        self.epsilon = epsilon
        self.use_l2 = use_l2
        self.restart_threshold = restart_threshold # Average usage threshold to consider a code 'dead'

        self.register_buffer('ema_cluster_size', torch.zeros(self.codebook_size))
        self.register_buffer('ema_w', torch.Tensor(self.codebook_size, self.code_size))
        self.ema_w.data.normal_() # Better init
        
        self.embedding.weight.requires_grad = False

    def forward(self, z_e_x):
        # Optional: L2 Normalize inputs and codebook (Spherical VQ)
        if self.use_l2:
            z_e_x_proc = F.normalize(z_e_x, p=2, dim=1)
            weight_proc = F.normalize(self.embedding.weight, p=2, dim=1)
        else:
            z_e_x_proc = z_e_x
            weight_proc = self.embedding.weight

        # 1. Find nearest neighbors (Cosine similarity if L2 is on)
        with torch.no_grad():
            # Standard distance calculation using processed (normalized) vectors
            codebook_sqr = torch.sum(weight_proc ** 2, dim=1)
            inputs_sqr = torch.sum(z_e_x_proc ** 2, dim=1, keepdim=True)
            distances = torch.addmm(codebook_sqr + inputs_sqr, z_e_x_proc, weight_proc.t(), alpha=-2.0, beta=1.0)
            indices = torch.argmin(distances, dim=1)
        
        # 2. Straight-through
        z_q_x = self.embedding(indices)
        if self.use_l2:
            z_q_x = F.normalize(z_q_x, p=2, dim=1) # Keep quantized vectors on sphere
        
        # 3. EMA & Random Restart (Only during training)
        if self.training:
            encodings = torch.zeros(indices.size(0), self.codebook_size, device=z_e_x.device)
            encodings.scatter_(1, indices.unsqueeze(1), 1)
            
            # Update EMA
            self.ema_cluster_size.data.mul_(self.decay).add_(1 - self.decay, encodings.sum(0))
            dw = torch.matmul(encodings.t(), z_e_x_proc)
            self.ema_w.data.mul_(self.decay).add_(1 - self.decay, dw)
            
            # Random Restart for dead codes
            # If a codebook vector is used much less than average, replace it with a random sample from batch
            usage = encodings.sum(0)
            dead_indices = (usage < self.restart_threshold).nonzero().view(-1)
            if len(dead_indices) > 0 and len(z_e_x) > len(dead_indices):
                # Pick random samples from current batch
                rand_idx = torch.randperm(len(z_e_x))[:len(dead_indices)]
                self.ema_w.data[dead_indices] = z_e_x_proc[rand_idx] * self.restart_threshold
                self.ema_cluster_size.data[dead_indices] = self.restart_threshold
            
            # Re-calculate weights
            n = torch.sum(self.ema_cluster_size.data)
            normalized_cluster_size = (
                (self.ema_cluster_size.data + self.epsilon)
                / (n + self.codebook_size * self.epsilon) * n)
            
            self.embedding.weight.data.copy_(self.ema_w.data / normalized_cluster_size.unsqueeze(1))

        # Commitment Loss
        # We always calculate loss against the encoder's raw output vs the quantized vector
        loss = self.beta * F.mse_loss(z_e_x_proc, z_q_x.detach(), reduction='none').sum(dim=1)
        
        return z_q_x, z_q_x, loss

    def compute_distances(self, inputs):
        """ Used by analysis scripts """
        if self.use_l2:
            inputs = F.normalize(inputs, p=2, dim=1)
            weight = F.normalize(self.embedding.weight, p=2, dim=1)
        else:
            weight = self.embedding.weight
            
        embedding_size = weight.size(1)
        inputs_flatten = inputs.view(-1, embedding_size)
        codebook_sqr = torch.sum(weight ** 2, dim=1)
        inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
        distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, weight.t(), alpha=-2.0, beta=1.0)
        return distances
