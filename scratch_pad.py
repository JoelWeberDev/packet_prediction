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


ts = (torch.ones((10, 20)), torch.zeros((20, 10)))


dc = deepcopy(ts)

print(dc)

print(np.random.choice(range(10), 3))
