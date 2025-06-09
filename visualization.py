"""
@Author: Joel Weber
@Date: 2025-06-03
@Description: Proceedures for analysis and visualization of our processed
data.

@Notes:

@TODO:
    - Header histogram
    - Simple numerical data point graphs
        - Line, bar, scatter, pie, ect.

"""

# Python imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Iterable

# Local imports
from CONSTANTS import *


### Plotting functions ###
def plot_bar(data, **kwargs):
    if isinstance(data, dict):
        assert all(
            [isinstance(val, NUMERICS) for val in data.values()]
        ), f"All values must be of numeric type"
        plt.bar(data.keys(), data.values(), **kwargs)
    elif isinstance(data, Iterable):
        data = np.array(data)
        if kwargs.get("height") is not None:
            # The user provided height so we are ok
            plt.bar(data, **kwargs)
        else:
            assert (
                len(data.shape) >= 2
            ), f"If no height is provided, the data must have at least 2 dimensions"
            plt.bar(data[0], data[1], **kwargs)
    else:
        raise Exception(f"Data type {type(data)} is not supported")

    plt.show()
