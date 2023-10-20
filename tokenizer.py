from collections import Counter
import os
import pickle
from typing import List

import numpy as np


class Tokenizer:
    def __init__(
        self,
        corpus_file_path="data/simple.txt",
        top_k=1000,
        mode="character"   # "character", "whitespace"
    ):
        self.top_k = top_k
        self.mode = mode
        # Load corpus file
        if not os.path.exists(corpus_file_path):
            raise IOError("=== Corpus file not found. ===")
        else:
            with open(f"{corpus_file_path}", encoding="utf-8-sig") as f_in:
                self.corpus = f_in.read()
        self.token_to_idx, self.idx_to_token = self.create_mapping_dict()
        self.vocab_size = len(self.token_to_idx)
    
    def tokenize(self, text):
        text = text.lower()
        if self.mode == "character":
            all_tokens = [char for sequence in text.split() for char in sequence]
        elif self.mode == "whitespace":
            all_tokens = text.split()
        return all_tokens
    
    def create_mapping_dict(self):
        token_to_idx, idx_to_token = {}, {}
        token_counter = Counter(self.tokenize(self.corpus))
        unique_tokens = list(token_counter.keys())[:self.top_k]
        unique_tokens = ["EOS", "UNK"] + unique_tokens
        for idx, token in enumerate(unique_tokens):
            token_to_idx[token] = idx
            idx_to_token[idx] = token
        return token_to_idx, idx_to_token
    
    def load_mapping_dict(self):
        pass
    
    def encode(self, token):
        return self.token_to_idx.get(token, self.token_to_idx["UNK"])
    
    def decode(self, idx):
        return self.idx_to_token.get(idx, "UNK")
    
    def encode_sequence(self, sequence: List[str]) -> List[int]:
        return [self.encode(token) for token in sequence]
    
    def decode_sequence(self, sequence: List[int]) -> List[str]:
        return [self.decode(idx) for idx in sequence]

    def one_hot_encode(self, idx):
        one_hot_vector = np.zeros(self.vocab_size)
        one_hot_vector[idx] = 1
        return one_hot_vector
    
    def one_hot_encode_sequence(self, sequence: List[str]):
        encoding = [self.encode(token) for token in sequence]
        encoding = [self.one_hot_encode(idx) for idx in encoding]
        return encoding

if __name__ == "__main__":
    tokenizer = Tokenizer()
    print(tokenizer.encode("UNK"))   # 1
    print(tokenizer.decode(0))   # "EOS"
    print(tokenizer.encode_sequence(["a", "b", "a", "EOS"]))   # [2, 3, 2, 0]
    print(tokenizer.decode_sequence([0, 1, 2, 3, 4, 5, 6]))   # ['EOS', 'UNK', 'a', 'b', 'UNK', 'UNK', 'UNK']
    print(tokenizer.one_hot_encode_sequence([0, 1, 2, 3, 2]))