#!/usr/bin/env python
"""
Loss functions for spatial-aware biomass prediction.

Author: najahpokkiri
Date: 2025-05-28
"""

import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    """Huber loss function that is less sensitive to outliers."""
    
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        abs_error = torch.abs(y_pred - y_true)
        quadratic = torch.min(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic
        return torch.mean(0.5 * quadratic.pow(2) + self.delta * linear)


class SpatialLoss(nn.Module):
    """Loss function that penalizes spatial autocorrelation in residuals."""
    
    def __init__(self, mse_weight=0.8, spatial_weight=0.2, device=None):
        super(SpatialLoss, self).__init__()
        self.mse_weight = mse_weight
        self.spatial_weight = spatial_weight
        self.device = device
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true, coordinates=None):
        # Standard MSE loss
        mse_loss = self.mse(y_pred, y_true)
        
        # If no coordinates provided, just return MSE
        if coordinates is None:
            return mse_loss
        
        # Calculate residuals
        residuals = y_pred - y_true
        
        # Convert coordinates to tensor if needed
        if not torch.is_tensor(coordinates):
            coordinates = torch.tensor(coordinates).float().to(self.device)
        
        # Calculate spatial weights (simplified)
        # For efficiency on larger batches, use a subsample if n is large
        n = len(residuals)
        if n > 100:
            indices = torch.randperm(n)[:100]
            sub_coords = coordinates[indices]
            sub_residuals = residuals[indices]
            n = 100
        else:
            sub_coords = coordinates
            sub_residuals = residuals
        
        # Calculate spatial penalty
        spatial_penalty = 0.0
        
        # For each point, find the nearest points and check residual similarity
        for i in range(n):
            # Calculate distances from point i to all other points
            dists = torch.sqrt(torch.sum((sub_coords - sub_coords[i]).pow(2), dim=1))
            
            # Get indices of nearest points (excluding self)
            _, indices = torch.topk(dists, min(11, n), largest=False)
            indices = indices[1:]  # Remove self
            
            # Calculate residual differences
            res_diffs = torch.abs(sub_residuals[indices] - sub_residuals[i])
            
            if len(indices) > 1:
                # Normalize distances to [0,1]
                norm_dists = dists[indices] / (torch.max(dists[indices]) + 1e-8)
                
                # Calculate penalty (closer points should have similar residuals)
                penalty = torch.mean(torch.abs(res_diffs - norm_dists))
                spatial_penalty += penalty
        
        spatial_penalty = spatial_penalty / n
        
        # Combine MSE and spatial penalty
        total_loss = self.mse_weight * mse_loss + self.spatial_weight * spatial_penalty
        
        return total_loss


def create_loss_function(config, device=None):
    """Create a loss function based on the configuration."""
    loss_type = config.loss_function.lower()
    
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "huber":
        return HuberLoss(delta=config.huber_delta)
    elif loss_type == "spatial":
        return SpatialLoss(mse_weight=1.0-config.spatial_loss_weight,
                          spatial_weight=config.spatial_loss_weight,
                          device=device)
    else:
        print(f"Unknown loss function: {loss_type}, defaulting to MSE")
        return nn.MSELoss()