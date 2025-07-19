"""
@Author: Joel Weber
@Date: 2025-07-01
@Description: Libary of helpful custom functions in training the LSTM model

@Notes:

@TODO:
"""

### Python imports ###
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Dict, Tuple, Iterator
from dataclasses import dataclass

### Local imports ###
from modules.CONSTANTS import *
from modules.custom_datasets import PacketDataset


### Custom data classes ###
@dataclass
class PacketIterator:
    packet_it: Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    n_packets: int
    cat_dims: List[int]
    num_dims: int

    def __next__(self):
        return next(self.packet_it)

    def __iter__(self):
        return self.packet_it

    def __len__(self):
        return self.n_packets


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


def hidden_state_contrast_loss(
    model, cat_features, num_features, payload
) -> torch.Tensor:
    """Force hidden states to be different for different inputs but consistent for the same input"""

    # First forward pass
    model.reset_hidden()
    logits1, preds1 = model(cat_features, num_features, payload)
    hidden1 = model.hidden.detach().clone()

    # Second forward pass with same input - should produce similar hidden state
    model.reset_hidden()
    logits2, preds2 = model(cat_features, num_features, payload)
    hidden2 = model.hidden.detach().clone()

    # Third forward pass with perturbed input - should produce different hidden state
    perturbed_cat = cat_features.clone()
    # Slightly change categorical features
    if random.random() > 0.5 and perturbed_cat.numel() > 0:
        idx = random.randint(0, perturbed_cat.numel() - 1)
        perturbed_cat.view(-1)[idx] = (
            perturbed_cat.view(-1)[idx] + 1
        ) % perturbed_cat.max()

    model.reset_hidden()
    _, _ = model(perturbed_cat, num_features, payload)
    hidden3 = model.hidden.detach().clone()

    # Calculate similarity between same inputs (should be high)
    same_sim = F.cosine_similarity(hidden1.view(-1), hidden2.view(-1), dim=0)

    # Calculate similarity between different inputs (should be low)
    diff_sim = F.cosine_similarity(hidden1.view(-1), hidden3.view(-1), dim=0)

    # We want same_sim to be high (close to 1) and diff_sim to be low
    return torch.tensor(
        max(1e-6, (1 - same_sim.item()) + max(0, diff_sim.item() - 0.5))
    )


def hidden_reliance_loss(
    model,
    criterion,
    cat_features,
    num_features,
    payload,
    h_loss,
    scale: float = 1.0,
    margin: float = HIDDEN_RELIANCE_MARGIN,
) -> torch.Tensor:
    """
    @Description: Computes a loss that punishes accurate results when the hidden state is None.
    This effectively punishes explicit sequence memorization and builds reliance on the hidden
    state.

    @Notes:
        - This uses a reflected scaled logistic function to invert the loss reward structure
        - desired: noh_loss - h_loss > margin

    @Returns:
    """
    # Get the loss when hidden state is reset
    orig_hidden = None
    if model.hidden is not None:
        orig_hidden = model.hidden.detach().clone()

    model.reset_hidden()
    noh_logits, _ = model(cat_features, num_features, payload)
    noh_loss = criterion(noh_logits, payload)

    # restore the hidden state
    model.reset_hidden()
    model.hidden = orig_hidden

    diff = noh_loss - h_loss

    # return torch.relu(margin - noh_loss + h_loss) * scale
    # return scale * torch.exp(-(diff - margin))
    return scale * (1 - torch.sigmoid((diff - margin) * 5))


def compute_hidden_state_regularization(hidden) -> torch.Tensor:
    """Penalize low variance in hidden states (static memory)"""
    if hidden is None:
        return torch.tensor(0.0, dtype=torch.float32, device=DEVICE)

    # Calculate variance across hidden dimensions
    mean_hidden = torch.mean(hidden, dim=2, keepdim=True)
    variance = torch.mean((hidden - mean_hidden).pow(2))

    # Penalize low variance (want dynamic, changing hidden states)
    return 0.1 * torch.exp(-variance * 5)


def diversity_loss(predictions: torch.Tensor, window_size: int = 5) -> torch.Tensor:
    """Penalize repetitive predictions within a sliding window"""
    if len(predictions) < window_size:
        return torch.tensor(0.0, device=predictions.device)

    loss = torch.tensor(0.0, device=predictions.device)
    count = 0
    for i in range(len(predictions) - window_size + 1):
        window = predictions[i : i + window_size]
        unique_tokens = len(torch.unique(window))
        # Penalize low diversity (fewer unique tokens)
        diversity_score = unique_tokens / window_size
        loss += (1 - diversity_score) ** 2
        count += 1

    return loss / max(1, count)


def entropy_regularization(
    logits: torch.Tensor, target_entropy: float = 2.0
) -> torch.Tensor:
    """Encourage higher entropy in predictions to prevent collapse"""
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    target = torch.full_like(entropy, target_entropy)
    return F.mse_loss(entropy, target)


def pattern_break_loss(hidden_states: List[torch.Tensor]) -> torch.Tensor:
    """Penalize similar hidden states across different time steps"""
    if len(hidden_states) < 2:
        device = hidden_states[0].device if hidden_states else torch.device("cpu")
        return torch.tensor(0.0, device=device)

    loss = torch.tensor(0.0, device=hidden_states[0].device)
    count = 0

    # Compare hidden states across time steps
    for i in range(len(hidden_states)):
        for j in range(
            i + 1, min(i + 5, len(hidden_states))
        ):  # Compare with next 4 states
            sim = F.cosine_similarity(
                hidden_states[i].view(-1), hidden_states[j].view(-1), dim=0
            )
            # Penalize high similarity
            loss += torch.relu(sim - 0.3)  # Only penalize if similarity > 0.3
            count += 1

    return loss / max(1, count)
