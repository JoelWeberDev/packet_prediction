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


t1 = torch.ones((10, 20))
t2 = torch.zeros((20, 10))

print(torch.cat([t1.flatten(), t2.flatten()], dim=-1))

t = torch.ones(10).unsqueeze(1).repeat(1, 10)

print(t)

