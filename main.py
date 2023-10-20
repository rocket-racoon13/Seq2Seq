import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split

from dataset import CustomDataset
from data_utils import generate_dataset, create_token_dictionary
from rnn import RNN
from tokenizer import Tokenizer
from trainer import Trainer
from tester import Tester


def config():
    args = argparse.ArgumentParser()
    args.add_argument("--seed", type=int, default=777)
    args.add_argument("--data_dir", type=str, default="data/simple.txt")
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--batch_size", type=int, default=10)
    args.add_argument("--hidden_size", type=int, default=128)
    args.add_argument("--learning_rate", type=float, default=1e-3)
    args.add_argument("--train", action="store_true")
    args.add_argument("--test", action="store_true")
    return args


if __name__ == "__main__":
    
    args = config()
    
    # Create random dataset
    if not os.path.exists(args.data_dir):
        generate_dataset()
    
    with open(args.data_dir, encoding="utf-8-sig") as f_in:
        data = f_in.read().splitlines()
    
    # Create tokenizer
    tokenizer = Tokenizer(
        corpus_file_path="data/simple.txt",
        top_k=1000,
        mode="character"
    )
    
    # Divide data into (character-level) inputs and targets
    inputs, targets = [], []
    inputs = [list(sequence) for sequence in data]
    targets = [list(sequence) + ["EOS"] for sequence in data]
    
    # One-hot encode inputs and targets
    inputs = [tokenizer.one_hot_encode_sequence(seq) for seq in input]
    targets = [tokenizer.one_hot_encode_sequence(seq) for seq in targets]
    for data in [inputs, targets]:
        data = np.array(data)
        data = data.reshape(data.shape[0], data.shape[1], 1)
    
    # Load and split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        inputs, targets, test_size=0.2, random_state=args.seed
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=args.seed
    )
    
    # Create dataset
    train_ds = CustomDataset(X_train, y_train)
    valid_ds = CustomDataset(X_valid, y_valid)
    test_ds = CustomDataset(X_test, y_test)
    
    # Model
    model = RNN(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        lr=args.learning_rate
    )
    
    if args.train:
        trainer = Trainer(
            args=args,
            train_ds=train_ds,
            valid_ds=valid_ds,
            model=model
        )
    
    if args.test:
        pass