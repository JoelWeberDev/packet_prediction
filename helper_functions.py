"""
@Author: Joel Weber
@Date: 2025-07-01
@Description: Libary of helpful custom functions in training the LSTM model

@Notes:

@TODO:
"""

### Python imports ###
import numpy as np
import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Iterator
from dataclasses import dataclass

### Local imports ###
from CONSTANTS import *
from preprocessing import load_df


### Custom data classes ###
@dataclass
class ConvResults:
    avg_loss: float
    avg_acc: float
    conv_loss: List[float]
    conv_acc: List[float]
    hidden_states: torch.Tensor | None
    cell_states: torch.Tensor | None


@dataclass
class EpochResults:
    avg_train_loss: float = float("inf")
    avg_train_acc: float = 0.0
    avg_val_loss: float = float("inf")
    avg_val_acc: float = 0.0


### Custom helper functions ###
def get_memory(device: str = DEVICE) -> Dict[str, float]:
    """
    @Description: gets the total memory usage in mb for the specified device

    @Notes:

    @Returns: dict of allocated and reserved memory
    """
    if device == "cuda":
        return {
            "allocated": torch.cuda.memory_allocated(device=device) / 1024**2,
            "reserved": torch.cuda.memory_reserved(device=device) / 1024**2,
            "max_reserved": torch.cuda.max_memory_reserved(device=device)
            / 1024**2,  # Peak usage
        }
    else:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "resident": memory_info.rss / 1024**2,  # Resident Set Size in MB
            "virtual": memory_info.vms / 1024**2,  # Virtual Memory Size in MB
        }


def print_update(**kwargs):
    print(f"\n")
    for key, val in kwargs.items():
        print(f"    {key}: {val}")

    mem_stats = get_memory()
    for key, value in mem_stats.items():
        print(f"    {key} memory: {value} MB")

    print()


def plot_metrics(
    loss_data: List[float] | np.ndarray,
    title: str | None = None,
    x_label: str = "Batch",
    y_label: str = "Batch loss",
):
    """
    @Description: Creates a line plot of the loss over time

    @Notes:

    @Returns:
    """
    plt.plot(loss_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title is None:
        title = f"{y_label} vs {x_label}"
    plt.title(title)
    plt.show()


def sample_with_temperature(logits: torch.Tensor, temp: float = 1.0) -> int:
    if temp == 0:
        return int(torch.argmax(logits, dim=-1))

    probs = F.softmax(logits / temp, dim=-1)
    return int(torch.multinomial(probs, 1).squeeze(-1))
