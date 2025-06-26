import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable


s = "Always work hard no matter the reward"
s2 = "The leaves that blow in the wind get nowhere"

l = list()

l = l + [0] * 10
print(l)

TRAIN_VAL_TEST_PERC = np.array([0.70, 0.15, 0.15])
TRAIN_VAL_TEST_PERC /= sum(TRAIN_VAL_TEST_PERC)

print(TRAIN_VAL_TEST_PERC)
print(np.sum(TRAIN_VAL_TEST_PERC))
