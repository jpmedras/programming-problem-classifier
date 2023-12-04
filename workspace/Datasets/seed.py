import torch
import random
import numpy as np

def define_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    #torch.use_deterministic_algorithms(True)