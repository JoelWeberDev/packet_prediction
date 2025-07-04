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

expnd = t.unsqueeze(0).expand(12, -1)

other = torch.zeros((12, 7), dtype=torch.long)

# print(t)

cmb = torch.cat([other, expnd], dim=-1)

print(cmb)
