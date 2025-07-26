import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from typing import Iterable, List
from dataclasses import dataclass
from modules.CONSTANTS import *
from copy import deepcopy


print(torch.sigmoid(torch.tensor(-10)))


t = torch.tensor(range(10)).tolist()

l = [v for v in t]
print(np.mean(l))
