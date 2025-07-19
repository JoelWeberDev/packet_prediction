import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Iterable, List
from dataclasses import dataclass
from modules.CONSTANTS import *
from copy import deepcopy


t = torch.tensor(range(10))


v = torch.tensor(range(10)).unsqueeze(1)

print(v[2:3])

t1 = torch.ones(7, dtype=torch.float32)
t2 = torch.tensor(range(7), dtype=torch.float32)

print(torch.dot(t1, t2))
print(t1.dot(t2))
