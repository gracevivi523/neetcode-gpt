import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        

        x      = np.array(x,      dtype=float)
        W1     = np.array(W1,     dtype=float)
        b1     = np.array(b1,     dtype=float)
        W2     = np.array(W2,     dtype=float)
        b2     = np.array(b2,     dtype=float)
        y_true = np.array(y_true, dtype=float)
        
                
        # forward pass
        z1 = np.dot (W1, x) + b1
        a1 = np.maximum(0, z1)
        y_pred = np.dot(a1, W2.T) + b2
        loss = ((y_pred - y_true) ** 2).mean()

        # ── Backward pass ─────────────────────────────────────────────────────
        n = len(y_true)   # number of output units (used in MSE normalisation)

        # dL/d(y_pred): derivative of MSE = 2*(y_pred - y_true)/n
        d_ypred = 2 * (y_pred - y_true) / n   # (out,)

        # Layer 2 gradients  (y_pred = a1 @ W2.T + b2)
        # dL/dW2 = d_ypred^T @ a1  →  outer product, shape (out, h)
        dW2 = np.outer(d_ypred, a1)            # (out, h)
        db2 = d_ypred                          # (out,)

        # Gradient flowing back into a1
        # d_ypred has shape (out,), W2 has shape (out, h)
        da1 = np.dot(d_ypred, W2)                # (h, out) @ (out,) = (h,)

        # Through ReLU: gate with (z1 > 0)
        dz1 = da1 * (z1 > 0)    # (h,)

        # Layer 1 gradients  (z1 = x @ W1 + b1)
        # dL/dW1 = x^T @ dz1  →  outer product, shape (in, h)
        dW1 = np.outer(dz1, x)                 # (in, h)
        dW1 += 0.0 # normalize dw1
        db1 = dz1                              # (h,)

        # ── Package results ───────────────────────────────────────────────────
        return {
            'loss': round(float(loss), 4),
            'dW1':  np.round(dW1, 4).tolist(),
            'db1':  np.round(db1, 4).tolist(),
            'dW2':  np.round(dW2, 4).tolist(),
            'db2':  np.round(db2, 4).tolist(),
        }

        


