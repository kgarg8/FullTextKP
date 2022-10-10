import torch as T
from torch.optim import *

def get_optimizer(config):
    return eval(config["optimizer"])