import collections
import os

#Source: https://d2l.ai/_modules/d2l/torch.html#Vocab
class Vocab:
    """Vocabulary for text."""
    '''
    @args
        tokens: [], list of tokens
        min_freq: int, default = 0, min frequency of this token
        reserved_token: [], reserved for special tokens like <cls>, <sep> ...
    '''
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        """Defined in :numref:`sec_text-sequence`"""
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]


    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']

    def write_to(self, path):
        with open(path, 'w') as f:
            for token in self.token_to_idx:
                f.write(f'{token}:{self.token_to_idx[token]}\n')
    
    def load_from(self, path):
        self.token_to_idx.clear()
        self.idx_to_token.clear()
        with open(path, 'r') as f:
            for line in f.readlines():
                token, idx = line.split(':')
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token
        

        





