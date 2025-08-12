import sys, os
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import higher
import subprocess
from typing import Iterable, List
from dataclasses import dataclass
from copy import deepcopy, copy


p = np.array(
    [
        [
            0.371,
            0.921,
            0.488,
            0.236,
            0.530,
            0.954,
            0.236,
            0.402,
            0.979,
            0.477,
        ]
    ]
)
p /= np.linalg.norm(p)
p = torch.tensor(p)

q = np.array(
    [
        0.879,
        0.340,
        0.988,
        0.126,
        0.699,
        0.042,
        0.476,
        0.414,
        0.316,
        0.399,
    ]
)
q /= np.linalg.norm(q)

criterion = nn.CrossEntropyLoss()

for i in range(10):
    guess = torch.tensor([i])
    loss = criterion.forward(p, guess)

    print(f"{i}: {loss.item()}")
