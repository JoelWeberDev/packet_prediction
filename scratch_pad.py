import sys, os
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import higher
import subprocess
from typing import Iterable, List
from dataclasses import dataclass
from copy import deepcopy, copy

# Local includes
from final_code.helper_functions import FocusLoss


t = torch.ones(18)
z = torch.zeros((18, 258))

print(z.squeeze(1).shape)
