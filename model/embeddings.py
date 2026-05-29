import numpy as np
from numpy.typing import NDArray


class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        # embeddings: (vocab_size, embed_dim) matrix
        # token_ids: 1D array of integer token IDs
        # Return the embedding vectors for the given token IDs
        # return np.round(your_answer, 5)

        # batch, embed_dim = embeddings.shape # 3, 2
        # print(batch, embed_dim)
        output = []

        for i in range(len(token_ids)):
            output.append(np.round(embeddings[token_ids[i]], 5))


        return output




