import argparse
from datetime import datetime
import os

from sklearn.model_selection import train_test_split
import torch

from dataset import CustomDataset
from model_utils import get_optimizer, get_scheduler
from models.seq2seq import Seq2Seq
from tokenizer import Tokenizer
from trainer import Trainer
from tester import Tester
from generator import Generator
from utils import *


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=777)
    
    parser.add_argument("--data_dir", type=str, default="data/opendict-korean-proverb.txt")
    parser.add_argument("--inference_data_dir", type=str, default="data/inference.txt")
    parser.add_argument("--save_dir", type=str, default=f"outputs/{datetime.now().strftime('%Y%m%d_%H-%M')}")
    parser.add_argument("--model_name", type=str)
    
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bidirectional", type=bool, default=False)
    parser.add_argument("--nonlinearity", type=str, default="tanh")
    parser.add_argument("--bias", type=bool, default=True)
    
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--scheduler", type=str, default="lambdaLR")
    
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--saving_steps", type=int, default=100)
    
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    
    args = parser.parse_args()
    return args


def main(args):
    
    # Get device
    device = get_device(args)
    print(f"=== Device Type: {device} ===")
    
    # Load dataset
    if not os.path.exists(args.data_dir):
        raise IOError("Dataset not found.")
    else:
        with open(args.data_dir, encoding="utf-8-sig") as f_in:
            data = f_in.read().splitlines()
    
    # Create output dir
    if args.train:
        os.makedirs(args.save_dir, exist_ok=True)
        log_dir = os.path.join(args.save_dir, "log")
        os.makedirs(log_dir, exist_ok=True)
    
    # Create tokenizer
    tokenizer = Tokenizer(
        args,
        top_k=1000,
        mode="whitespace"
    )
    input_size = tokenizer.vocab_size
    
    # Create inputs and targets for predicting the following token
    # from the previous sequence of tokens
    inputs, targets = [], []
    for sequence in data:
        tokens = sequence.split() + ["EOS"]
        for i in range(1, len(tokens)):
            input = tokens[:i]
            target = tokens[i]
            inputs.append(input)
            targets.append(target)
    print("Sample data:", inputs[-1], targets[-1])
    
    # Apply encoding to inputs and targets
    inputs = [tokenizer.encode_sequence(seq) for seq in inputs]
    targets = tokenizer.encode_sequence(targets)
    print("Sample encoding:", inputs[-1], targets[-1])
    
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
    
    # Create model, optimizer, scheduler
    model = Seq2Seq(args, input_size)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    
    # Load trained model
    if args.save_dir and args.model_name:
        ckpt = torch.load(
            os.path.join(args.save_dir, args.model_name),
            map_location = device
        )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        steps = ckpt["steps"]
        print(f"=== {args.model_name} -> LOAD COMPLETE ===")
    
    # Train
    if args.train:
        trainer = Trainer(
            args=args,
            train_ds=train_ds,
            valid_ds=valid_ds,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        trainer.train()
    
    # Test
    if args.test:
        tester = Tester(
            args=args,
            test_ds=test_ds,
            model=model,
            device=device
        )
        tester.test()
        
    # Generate
    if args.generate:
        generator = Generator(
            args=args,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )
        generator.generate()


if __name__ == "__main__":
    args = config()
    main(args)