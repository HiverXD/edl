# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class SpectralKMeansManager:
    """
    Manages various spectral clustering algorithms (Bisecting, Weighted, etc.)
    using Laplacian embeddings psi(s).
    """
    def __init__(self, calc, n_clusters=10):
        self.calc = calc
        self.n_clusters = n_clusters
        
    def get_kde_weights(self, data, bandwidth=0.1):
        """
        Calculates weights as 1/density using Kernel Density Estimation.
        """
        print("Calculating KDE weights...")
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)
        log_density = kde.score_samples(data)
        density = np.exp(log_density)
        
        # Avoid division by zero and extreme values
        weights = 1.0 / (density + 1e-6)
        # Normalize weights so they sum to N
        weights = weights / np.mean(weights)
        return weights

    def run_bisecting_kmeans(self, data, weights=None):
        """
        Hierarchical Bisecting K-means. 
        Splits the cluster with largest inertia into two until n_clusters is reached.
        """
        print("Running Bisecting K-means (Weighted: {0})...".format(weights is not None))
        
        # Start with all data in one cluster
        clusters = [np.arange(len(data))]
        
        while len(clusters) < self.n_clusters:
            # Find the cluster with the highest inertia
            inertias = []
            for cluster_indices in clusters:
                if len(cluster_indices) < 2:
                    inertias.append(-1)
                    continue
                    
                cluster_data = data[cluster_indices]
                cluster_weights = weights[cluster_indices] if weights is not None else None
                
                # Calculate inertia for this cluster
                km = KMeans(n_clusters=1, n_init=1).fit(cluster_data, sample_weight=cluster_weights)
                inertias.append(km.inertia_)
                
            # Pick the cluster to split
            idx_to_split = np.argmax(inertias)
            cluster_to_split = clusters.pop(idx_to_split)
            
            # Split it into 2 using K-means
            cluster_data = data[cluster_to_split]
            cluster_weights = weights[cluster_to_split] if weights is not None else None
            
            km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(cluster_data, sample_weight=cluster_weights)
            labels = km.labels_
            
            # Add new sub-clusters
            clusters.append(cluster_to_split[labels == 0])
            clusters.append(cluster_to_split[labels == 1])
            
        # Convert back to flat labels
        final_labels = np.zeros(len(data), dtype=int)
        centroids_psi = []
        for i, cluster_indices in enumerate(clusters):
            final_labels[cluster_indices] = i
            # Mean position in embedding space (weighted if available)
            if weights is not None:
                w_sum = np.sum(weights[cluster_indices])
                centroid = np.sum(data[cluster_indices] * weights[cluster_indices][:, np.newaxis], axis=0) / w_sum
            else:
                centroid = np.mean(data[cluster_indices], axis=0)
            centroids_psi.append(centroid)
            
        return final_labels, np.array(centroids_psi)

    def run_weighted_kmeans(self, data, weights):
        """
        Standard K-means with KDE weights.
        """
        print("Running Weighted K-means...")
        km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        km.fit(data, sample_weight=weights)
        
        return km.labels_, km.cluster_centers_

    def get_centroids_in_state_space(self, data_psi, data_s, centroids_psi):
        """
        Finds the nearest neighbor in actual state space for each psi-space centroid.
        """
        centroids_s = []
        for c_psi in centroids_psi:
            dists = np.linalg.norm(data_psi - c_psi, axis=1)
            nn_idx = np.argmin(dists)
            centroids_s.append(data_s[nn_idx])
        return np.array(centroids_s)
