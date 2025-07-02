import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Iterable
from dataclasses import dataclass
from CONSTANTS import *


t = torch.tensor(list(range(10)), dtype=torch.long)

embedder = nn.Embedding(VOCAB_DIM, BYTE_EMBED_DIM)

bytes = torch.tensor(
    [
        256,
        0,
        4,
        77,
        81,
        84,
        84,
        4,
        2,
        0,
        60,
        0,
        23,
        48,
        56,
        48,
        98,
        52,
        54,
        49,
        56,
        55,
        97,
        50,
        102,
        52,
        100,
        51,
        50,
        98,
        53,
        49,
    ],
    dtype=torch.long,
)

embs = embedder(bytes).reshape(1, -1)

print(embs.shape)

print(t.tolist())

l = [1, 2, 3]
l2 = list()

l2.append(torch.tensor(l))

l.append(4)

print(l2[0])
