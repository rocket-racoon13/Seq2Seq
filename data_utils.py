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


def one_hot_encode(idx, vocab_size):
    one_hot_vector = np.zeros(vocab_size)
    one_hot_vector[idx] = 1
    return one_hot_vector


def one_hot_encode_sequence(sequence, token_to_idx, vocab_size):
    encoding = np.array([one_hot_encode(token_to_idx[token], vocab_size) for token in sequence])
    encoding = encoding.reshape(encoding.shape[0], encoding.shape[1], 1)
    return encoding


sequences = generate_dataset()
token_to_idx, idx_to_token, num_sentences, vocab_size = sequences_to_dicts(sequences)
encoding = one_hot_encode_sequence(sequences[0], token_to_idx, vocab_size)
print(encoding.shape) # (15, 4, 1)