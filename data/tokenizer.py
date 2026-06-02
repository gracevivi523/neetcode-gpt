from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            counts = defaultdict(int)

            for i in range(len(tokens) - 1):
                counts[tokens[i], tokens[i+1]] += 1

            # print(counts)
            # if the corpus is a single token, there are no pairs, stop early.
            if not counts:
                break

            best = min(counts, key=lambda p: (-counts[p], p))
            # print(best)
            merges.append(list(best))

            merged, new_tokens, i = best[0] + best[1], [], 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return merges


