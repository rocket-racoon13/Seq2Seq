import argparse
import os

from data_utils import generate_dataset, create_token_dictionary
from tokenizer import Tokenizer


def config():
    args = argparse.ArgumentParser()
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--batch_size", type=int, default=10)
    args.add_argument("--train", action="store_true")
    args.add_argument("--test", action="store_true")
    return args


if __name__ == "__main__":
    
    args = config()
    
    # Create random dataset
    if not os.path.exists("data/simple.txt"):
        generate_dataset()
    
    # Create tokenizer
    if not os.path.exists("dictionary/simple.pkl"):
        create_token_dictionary()
    else:
        tokenizer = Tokenizer("dictionary/simple.pkl")