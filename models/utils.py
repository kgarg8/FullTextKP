import math

import torch as T
import torch.nn as nn

def glorot_uniform_init(weight, fan_in, fan_out):
    v = 6 if (fan_in != 0 and fan_out != 0) else 3
    bound = float(math.sqrt(v/ (fan_in + fan_out)))
    nn.init.uniform_(weight, a=-bound, b =bound)