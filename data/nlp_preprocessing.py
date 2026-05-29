import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        all_sentences = positive + negative
        print(all_sentences)

        # 1. Build vocabulary
        all_words = []
        for sentence in all_sentences:
            for word in sentence.split():
                all_words.append(word)
        vocab = sorted(set(all_words))
        word_to_id = {word: i+1 for i, word in enumerate(vocab)}

        # 2. Encode each senetence
        my_tensors = []
        for sentence in all_sentences:
            my_tensor = []
            for word in sentence.split():
                value = word_to_id[word]
                my_tensor.append(value)
            my_tensors.append(torch.tensor(my_tensor, dtype=torch.float) )

        # 3. return
        return nn.utils.rnn.pad_sequence(my_tensors, batch_first=True, padding_value=0)        
