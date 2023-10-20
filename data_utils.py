from collections import Counter
import numpy as np

np.random.seed(42)


def generate_dataset(num_sequences=100):
    samples = []
    for _ in range(num_sequences):
        num_tokens = np.random.randint(1, 10)
        sample = ["a"] * num_tokens + ["b"] * num_tokens + ["EOS"]
        samples.append(sample)
    return samples


def sequences_to_dicts(sequences):
    all_tokens = [token for seq in sequences for token in seq]
    token_counter = Counter(all_tokens)
    unique_tokens = list(token_counter.keys())
    unique_tokens.append("UNK")
    
    num_sentences = len(sequences)
    vocab_size = len(unique_tokens)
    
    token_to_idx, idx_to_token = {}, {}
    for idx, token in enumerate(unique_tokens):
        token_to_idx[token] = idx
        idx_to_token[idx] = token
    
    return token_to_idx, idx_to_token, num_sentences, vocab_size


sequences = generate_dataset()
print(sequences[0])
sequences_to_dicts(sequences)