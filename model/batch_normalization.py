import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists

        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)
        beta = np.array(beta, dtype=float)
        running_mean = np.array(running_mean, dtype=float)
        running_var = np.array(running_var, dtype=float)
        
        batch_size, features = x.shape

        if training:
            ub = np.mean(x, axis=0)
            qb = np.mean((x - ub) ** 2, axis=0)
            x_hat = (x - ub) / np.sqrt(qb + eps)

            running_mean = np.round((1 - momentum) * running_mean + momentum * ub, 4)
            running_var = np.round((1 - momentum) * running_var + momentum * qb, 4)  # qb not qb[j]
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)

        y = np.round(gamma * x_hat + beta, 4)

        return y.tolist(), running_mean.tolist(), running_var.tolist()        

