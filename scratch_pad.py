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


ts = torch.stack([torch.tensor(range(i, i + 10)) for i in range(7)])

print(torch.sum(ts, dim=0))

w = torch.tensor(range(7))
print(w.unsqueeze(0).reshape(7, -1) * ts)

print(ts.shape)
