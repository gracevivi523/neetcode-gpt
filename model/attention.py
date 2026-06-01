import torch
import torch.nn as nn
from torchtyping import TensorType
import math

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.attention_dim = attention_dim
        self.key   = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        K = self.key(embedded)
        Q = self.query(embedded)
        V = self.value(embedded)

        # attention score
        scores =  Q @ K.transpose(1, 2) / math.sqrt(self.attention_dim)

        # causal mask
        seq_len = embedded.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=embedded.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        scores = torch.softmax(scores, dim=2)

        return (scores @ V).round(decimals=4)
