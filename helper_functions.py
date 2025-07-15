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
from custom_datasets import PacketDataset


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


def sample_with_temperature(logits: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    if temp == 0:
        return torch.argmax(logits, dim=-1)

    probs = F.softmax(logits / temp, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def google_get_embedding_dim(n_cats: int) -> int:
    # Google's categorical embedding formuala
    return min(MAX_CAT_EMB, round((n_cats * CAT_EMB_SCALAR) ** CAT_EMB_EXPO))


### Cross function variables ###
conv_list = list()
split_dict = {
    "train": [],
    "val": [],
    "test": [],
}

### Helper functions ###
def update_split_dict():
    global split_dict, conv_list
    n_convs = len(conv_list)
    n_in_dict = np.sum([len(elem) for elem in split_dict.values()])
    # Number of values to add for each category
    n_train = int(TRAIN_VAL_TEST_PERCS[0] * n_convs) - len(split_dict["train"])
    n_val = int(TRAIN_VAL_TEST_PERCS[1] * n_convs) - len(split_dict["val"])
    n_test = n_convs - n_val - n_train - len(split_dict["test"])

    # Now get how many are already in the split dict
    new_conv_nums = list(range(n_in_dict, n_in_dict + n_train + n_val + n_test))
    for key, n_nums in zip(split_dict.keys(), (n_train, n_val, n_test)):
        for _ in range(n_nums):
            # We randomly choose which category each conversation nubmer should go to
            conv_num = new_conv_nums.pop(np.random.randint(0, len(new_conv_nums)))
            split_dict[key].append(conv_num)

    assert (
        len(new_conv_nums) == 0
    ), f"Remaining converstion number list length must be 0, not {len(new_conv_nums)}"

def split_convs(conv_dfs: List[PacketDataset]) -> Dict[str, List[PacketDataset]]:
    # update the splti dict
    update_split_dict()

    ret = {"train": [], "val": [], "test": []}
    # Now use the indicies to split the conversations into train, validation, and test
    # We assume that each conv_df has one and only one conversation number
    for conv_df in conv_dfs:
        for key, conv_nums in split_dict.items():
            if conv_df.conv_num in conv_nums:
                ret[key].append(conv_df)

    return ret


    
### Custom loss functions ###
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = P_SMOOTHING):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.log_softmax(dim=-1)  # smooth winner takes all PDF
        n_classes = pred.size(dim=-1)
        true_dist = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        true_dist = true_dist * (1 - self.smoothing) + self.smoothing / n_classes
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))