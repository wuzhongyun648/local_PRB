import random

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_tensor(array):
    return torch.as_tensor(array, dtype=torch.float32, device=device)
