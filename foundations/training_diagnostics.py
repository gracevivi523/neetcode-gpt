import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        hooks = []

        def make_hook():
            def hook(module, input, output):
                acts = output.detach()
                acts_2d = acts.view(acts.shape[0], -1)          # (batch, neurons)
                stats.append({
                    'mean': round(acts.mean().item(), 4),
                    'std': round(acts.std().item(), 4),
                    'dead_fraction': round((acts_2d <= 0).all(dim=0).float().mean().item(), 4),
                })
            return hook

        for module in model.modules():
            # print(module)
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(make_hook()))

        with torch.no_grad():
            model(x)

        for h in hooks:
            h.remove()

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()

        loss = nn.MSELoss()(model(x), y)
        loss.backward()

        stats = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad.detach()
                stats.append({
                    'mean': round(grad.mean().item(), 4),
                    'std': round(grad.std().item(), 4),
                    'norm': round(grad.norm().item(), 4),
                })

        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        if any(s['dead_fraction'] > 0.5 for s in activation_stats):
            return 'dead_neurons'
        if any(s['norm'] > 1000.0 for s in gradient_stats):
            return 'exploding_gradients'
        if any(s['norm'] < 1e-5 for s in gradient_stats):
            return 'vanishing_gradients'
        if any(s['std'] < 0.1 for s in activation_stats):
            return 'vanishing_gradients'
        if any(s['std'] > 10.0 for s in activation_stats):
            return 'exploding_gradients'            
        return 'healthy'
