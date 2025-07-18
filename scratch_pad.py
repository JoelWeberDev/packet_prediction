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


tens = [torch.tensor(range(i, i + 10)) for i in range(5)]

t = torch.stack(tens)
print(t)
