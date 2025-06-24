import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Iterable


class HistoryQueue:
    """
    @Description: Stores the packet history for the model training in a memory
    safe way that drops the oldest packet once the max size is exceeded. Internally
    this is implemented using a circular structure and start / end pointers

    @Notes:
        - Removal is not implemented because it is not needed

    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data = list()
        self.first = 0  # first in
        self.last = 0  # last in

    def append(self, value):
        if len(self) == self.max_size:
            # replace the value at start
            self.data[self.first] = value

            self.first = self.inc_ptr(self.first)
            self.last = self.inc_ptr(self.last)

        else:
            self.data.append(value)
            self.last = self.inc_ptr(self.last)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slicing (e.g., queue[1:5])
            start = idx.start if idx.start is not None else 0
            start = max(-1 * len(self), start)
            start = min(len(self), start) % len(self)

            stop = idx.stop if idx.stop is not None else len(self)
            stop = max(-1 * len(self), stop)
            stop = min(len(self), stop) % (len(self) + 1)

            step = idx.step if idx.step is not None else 1
            return [
                self.data[(self.first + i) % len(self)]
                for i in range(start, stop, step)
            ]
        else:
            # Handle single index (e.g., queue[1])
            return self.data[(self.first + idx) % len(self)]

    ### Helper functions ###
    def inc_ptr(self, val: int) -> int:
        return (val + 1) % len(self)


hq = HistoryQueue(10)

for i in range(15):
    hq.append(i)


print(hq[len(hq) - 5 : -2])
l = list(range(10))

