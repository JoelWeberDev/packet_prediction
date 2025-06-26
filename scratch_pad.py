import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Iterable


t1 = torch.tensor(list(range(10)))
t2 = torch.tensor(list(range(10, 20)))

# for a, b in zip(t1, t2):
#     print(a, b)

t = torch.tensor([[0] * 10 for i in range(10)])

print(t)
