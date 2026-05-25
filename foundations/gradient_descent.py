import numpy as np
class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        # 1. Define Function and Gradient
        def f(x):
            return x**2

        def df(x):
            return 2*x

        # 2. Parameters
        x = init  # Starting point

        # 3. Gradient Descent Loop
        for i in range(iterations):
            derivative = df(x)
            x = x - learning_rate * derivative

        return round(x, 5)