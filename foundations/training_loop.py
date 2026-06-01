import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        n, f = X.shape
        w = np.zeros(f)
        b = 0.0


        for _ in range(epochs):
            y_hat = X @ w + b

            w -= lr * (2 / n) * X.T @ (y_hat - y)
            b -= lr * (2 / n) * (y_hat - y).sum()

        return (np.round(w, 5), round(b, 5))
            
