import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable


val = "0c"

print(int(val, base=16))

i = 0

test_bytes = (
    "102300044d5154540402003c00173237333763393931646665313433336538663735646433"
)

bt_len = int(len(test_bytes) / 2)
pl_len = 35
print(test_bytes[-2 * pl_len :]) 

l = list(range(7))

print(l[-3:])
