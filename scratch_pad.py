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
