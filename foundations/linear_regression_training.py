import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)

        gradients = np.zeros_like(initial_weights)
        predictions = self.get_model_prediction(X, initial_weights)

        for i in range(len(initial_weights)):
            gradients[i] = self.get_derivative(predictions, Y, len(X), X, i)
        
        updated_weights = initial_weights.copy()

        for _ in range(num_iterations):
            updated_weights -= self.learning_rate * gradients
            predictions = self.get_model_prediction(X, updated_weights)

            for i in range(len(initial_weights)):
                gradients[i] = self.get_derivative(predictions, Y, len(X), X, i)

        return np.round(updated_weights, 5)



