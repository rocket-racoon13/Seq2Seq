from collections import Counter
import pickle

import numpy as np

np.random.seed(42)


def generate_dataset(num_sequences=100):
    """
    동일 개수의 a, b로 이루어진 문자열 시퀀스(예: "aaaabbbb")를
    하나의 데이터로 하는 랜덤 데이터셋을 생성하고 저장하는 함수
    Args:
        num_sequences (int): Data size. Defaults to 100.
    """
    samples = []
    for _ in range(num_sequences):
        num_tokens = np.random.randint(1, 10)
        sample = "a" * num_tokens + "b" * num_tokens
        samples.append(sample)
    
    with open("data/simple.txt", "w", encoding="utf-8-sig") as f_out:
        f_out.write("\n".join(samples))


def create_token_dictionary(
        in_file_path="data/simple.txt",
        out_file_path="dictionary/simple.pkl"
    ):
    """
    데이터셋을 input으로 받아 character 단위 토큰에 대해
    토큰-id, id-토큰 매핑 딕셔너리 쌍을 만드는 함수
    Args:
        in_file_path (str, optional): Dataset file path. Defaults to "data/simple.txt".
        out_file_path (str, optional): Dictionary file path. Defaults to "dictionary/simple.pkl".
    """
    token_dict = {
        "vocab_size": 0,
        "token_to_idx": {},
        "idx_to_token": {}
    }
    with open(f"{in_file_path}", encoding="utf-8-sig") as f_in:
        data = f_in.read().splitlines() # list of str
        data = [list(sequence) + ["EOS"] for sequence in data]
        all_tokens = [token for sequence in data for token in sequence]
        token_counter = Counter(all_tokens)
        unique_tokens = list(token_counter.keys())
        unique_tokens.append("UNK")

        vocab_size = len(unique_tokens)
        
        token_to_idx, idx_to_token = {}, {}
        for idx, token in enumerate(unique_tokens):
            token_to_idx[token] = idx
            idx_to_token[idx] = token
            
        token_dict["vocab_size"] = vocab_size
        token_dict["token_to_idx"] = token_to_idx
        token_dict["idx_to_token"] = idx_to_token
        
    with open(f"{out_file_path}", "wb") as f_out:
        pickle.dump(token_dict, f_out)


def one_hot_encode(idx, vocab_size):
    one_hot_vector = np.zeros(vocab_size)
    one_hot_vector[idx] = 1
    return one_hot_vector


def one_hot_encode_sequence(sequence, token_to_idx, vocab_size):
    encoding = np.array([one_hot_encode(token_to_idx[token], vocab_size) for token in sequence])
    encoding = encoding.reshape(encoding.shape[0], encoding.shape[1], 1)
    return encoding