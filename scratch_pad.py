import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Iterable


t = torch.stack([torch.tensor(list(range(i, i + 10))) for i in range(10)])

print(t.shape)
