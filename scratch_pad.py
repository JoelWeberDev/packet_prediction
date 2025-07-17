import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Iterable, List
from dataclasses import dataclass
from CONSTANTS import *
from copy import deepcopy


l = torch.tensor(range(10))

print(l)

print(torch.stack([l, torch.tensor(range(5, 15))], dim=0).unsqueeze(0).shape)

t = torch.cat([l, torch.tensor([0] * 3)], dim=-1)

print(t.reshape(-1))
