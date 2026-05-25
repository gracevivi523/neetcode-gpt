import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        # Computing element wise exponential value
        print(np.max(z))
        exp_values = np.exp(z - np.max(z))

        # Computing sum of these values
        exp_values_sum = np.sum(exp_values)

        # Returing the softmax output.
        return np.round(exp_values/exp_values_sum,4)
