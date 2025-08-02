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
from modules.CONSTANTS import *
from copy import deepcopy
from modules.helper_functions import (
    conversation_tradjectory_loss,
    progressive_loss,
    conversation_trajectory_loss_simple,
    get_memory,
    get_git_hash,
)


print(get_git_hash())
