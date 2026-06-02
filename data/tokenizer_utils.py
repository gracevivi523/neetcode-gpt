from typing import List, Dict

class Solution:

    def _greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0

        while i < len(text):
            longest = None
            for length in range(1, len(text) - i + 1):
                candidate = text[i:i+length]
                if candidate in vocab:
                    longest = candidate

            if longest is None:
                longest = text[i]
            tokens.append(longest)

            i += len(str(longest))

        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split

        return [ self._greedy_tokenize(str(n), vocab) for n in numbers ]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        return len(self._greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        word_count = len(text.split(" "))
        token_count = self.count_tokens(text, vocab)
        return round(token_count / word_count, 4)
