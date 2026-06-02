from collections import defaultdict
from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        chars = set(text)
        chars = sorted(chars)

        stoi, itos = defaultdict(int), defaultdict(str)
        
        for i, c in enumerate(chars):
            stoi[c] = i
            itos[i] = c
        
        return (stoi, itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        result = []
        for t in text:
            result.append(stoi[t])
        return result

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        str = ''
        for i in ids:
            str += itos[i]

        return str
