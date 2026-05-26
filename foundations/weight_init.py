import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list

        torch.manual_seed(0)
        std = (2.0 / (fan_in + fan_out)) ** 0.5
        weights = torch.normal(mean=0.0, std=std, size=(fan_out, fan_in))
        return [[round(val, 4) for val in row] for row in weights.tolist()]

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = (2.0 / fan_in) ** 0.5
        weights = torch.normal(mean=0.0, std=std, size=(fan_out, fan_in))
        return [[round(val, 4) for val in row] for row in weights.tolist()]

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        
        # Generate all weights first
        weights = []
        current_dim = input_dim
        for _ in range(num_layers):
            if init_type == "xavier":
                std = (2.0 / (current_dim + hidden_dim)) ** 0.5
            elif init_type == "kaiming":
                std = (2.0 / current_dim) ** 0.5
            elif init_type == "random":
                std = 1.0
            
            W = torch.randn(hidden_dim, current_dim) * std
            weights.append(W)
            current_dim = hidden_dim
        
        # Generate input AFTER weights
        x = torch.randn(input_dim)
        
        stds = []
        for W in weights:
            x = torch.relu(x @ W.T)
            stds.append(round(x.std().item(), 2))
        
        return stds
