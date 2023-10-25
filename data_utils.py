import copy
import numpy as np

np.random.seed(42)


def pad_sequences(sequences, max_len=None, padding="pre"):
    if max_len is None:
        longest_len = max([len(seq) for seq in sequences])
        maxlen = longest_len
    else:
        maxlen = max_len
    
    padded = [0] * maxlen
    padded_sequences = []
    for seq in sequences:
        padded_seq = copy.deepcopy(padded)
        if padding == "pre":
            padded_seq[-len(seq):] = seq
        elif padding == "post":
            padded_seq[:len(seq)] = seq
        padded_sequences.append(padded_seq)
    return padded_sequences