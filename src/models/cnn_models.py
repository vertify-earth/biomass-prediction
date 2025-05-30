#!/usr/bin/env python
"""
CNN model architectures for spatial-aware biomass prediction.

Author: najahpokkiri
Date: 2025-05-28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNCoordinateModel(nn.Module):
    """CNN with coordinate channels for spatial awareness using InstanceNorm."""
    
    def __init__(self, input_channels, height, width):
        super(CNNCoordinateModel, self).__init__()
        
        # Account for 2 additional coordinate channels
        self.input_channels = input_channels + 2
        
        # Enhanced CNN architecture using InstanceNorm instead of BatchNorm
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm2d(128, affine=True)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
        # Calculate size after convolutions and pooling
        conv_height = height // 4
        conv_width = width // 4
        
        # Adjust if dimensions get too small
        if conv_height < 1: conv_height = 1
        if conv_width < 1: conv_width = 1
            
        # Fully connected layers with Layer Normalization
        self.fc1 = nn.Linear(128 * conv_height * conv_width, 128)
        self.norm5 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.norm6 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Create coordinate channels
        x_coords = torch.linspace(-1, 1, width).view(1, 1, 1, width).repeat(batch_size, 1, height, 1).to(x.device)
        y_coords = torch.linspace(-1, 1, height).view(1, 1, height, 1).repeat(batch_size, 1, 1, width).to(x.device)
        
        # Concatenate coordinate channels
        x = torch.cat([x, x_coords, y_coords], dim=1)
        
        # Convolutional layers
        x1 = self.relu(self.norm1(self.conv1(x)))
        x1 = self.pool(x1)
        x1 = self.dropout(x1)
        
        x2 = self.relu(self.norm2(self.conv2(x1)))
        x2 = self.pool(x2)
        x2 = self.dropout(x2)
        
        x3 = self.relu(self.norm3(self.conv3(x2)))
        x3 = self.dropout(x3)
        
        # Flatten
        x = x3.view(x3.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.norm5(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.norm6(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x.squeeze(1)


def create_model(model_type, input_channels, height, width, device):
    """Create a model based on the specified type."""
    print(f"\nCreating {model_type.upper()} model...")
    
    if model_type.lower() == "cnn_coordinate":
        model = CNNCoordinateModel(input_channels, height, width).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model