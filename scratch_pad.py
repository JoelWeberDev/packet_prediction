import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Iterable
from dataclasses import dataclass


t = torch.tensor(list(range(5, 10)) + list(range(3, 9)), dtype=torch.float32)
print(t)


pred = int(t.argmax(dim=-1))
mask = torch.ones_like(t, dtype=torch.bool)
mask[pred] = False
pred2 = t[mask].argmax(-1)

# print(t.std())
# print(f"pred1: {pred}, pred2: {pred2}, {t.mean()}, {t.median()}")

# Create example tensor
t = torch.tensor([5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8])


plt.plot(t)
plt.ylabel("loss")
plt.show()

l = list()

print(np.mean(l))

