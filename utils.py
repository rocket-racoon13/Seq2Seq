import random

import torch
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def get_device(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    return device