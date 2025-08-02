"""
@Author: Joel Weber
@Date: 2025-07-01
@Description: Libary of helpful custom functions in training the LSTM model

@Notes:

@TODO:
"""

### Python imports ###
import sys, os
import subprocess
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Dict, Tuple, Iterator, Optional, Any
from dataclasses import dataclass, field

### Local imports ###
from modules.CONSTANTS import *
from modules.custom_datasets import PacketDataset, ParsedPacket
from modules.preprocessing import load_dfs_from_dir, conversation_filter, col_values_set


### Custom data classes ###
@dataclass
class PacketIterator:
    packet_it: Iterator[ParsedPacket]
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
class PacketPrediction:
    logits: torch.Tensor
    preds: torch.Tensor


@dataclass
class ConvResults:
    packet_losses: List[torch.Tensor]
    packet_accs: List[float]
    conv_loss: float = float("inf")

    ### Getters ###
    @property
    def avg_packet_loss(self) -> float:
        return (
            float(np.mean([v.item() for v in self.packet_losses]))
            if len(self.packet_losses) > 0
            else float("inf")
        )

    @property
    def avg_packet_acc(self) -> float:
        return (
            float(np.mean(self.packet_accs))
            if len(self.packet_accs) > 0
            else float("inf")
        )

    ### Generics ###
    def __len__(self):
        return len(self.packet_accs)

    def __str__(self) -> str:
        return f"""
            avg packet loss: {self.avg_packet_loss}
            avg packet acc: {self.avg_packet_acc}
            conv loss: {self.conv_loss}
            num packets: {len(self)}
        """


@dataclass
class EpochResults:
    conv_results: List[ConvResults] = field(default_factory=list)
    max_conv_len: int = 0

    ### Getters ###
    @property
    def conv_packet_losses(self) -> List[float]:
        losses = list()
        for conv_res in self.conv_results:
            losses.append(conv_res.avg_packet_loss)

        return losses

    @property
    def conv_packet_accs(self) -> List[float]:
        accs = list()
        for conv_res in self.conv_results:
            accs.append(conv_res.avg_packet_acc)

        return accs

    @property
    def avg_packet_loss(self) -> float:
        conv_losses = self.conv_packet_losses
        return float(np.mean(conv_losses)) if len(conv_losses) > 0 else float("inf")

    @property
    def avg_packet_acc(self) -> float:
        conv_accs = self.conv_packet_accs
        return float(np.mean(conv_accs)) if len(conv_accs) > 0 else float("inf")

    @property
    def n_convs(self) -> int:
        return len(self.conv_results)

    @property
    def avg_conv_loss(self) -> float:
        # This loss measure's how well the model converges to a solution
        conv_losses = [conv_result.conv_loss for conv_result in self.conv_results]
        return float(np.mean(conv_losses)) if len(conv_losses) > 0 else float("inf")

    ### Generics ###
    def __len__(self):
        n_packets = 0
        for conv in self.conv_results:
            n_packets += len(conv)
        return n_packets

    def __str__(self):
        return f"""
            epoch avg packet loss: {self.avg_packet_loss}
            epoch avg packet acc: {self.avg_packet_acc}
            epoch avg conv loss: {self.avg_conv_loss}
            epoch tot num packets: {len(self)}
            epoch num convs: {self.n_convs}
        """


@dataclass
class ModelMetrics:
    epoch_results: List[EpochResults] = field(default_factory=list)

    # Properties #
    @property
    def epoch_avg_losses(self):
        return [epoch_result.avg_packet_loss for epoch_result in self.epoch_results]

    @property
    def epoch_avg_accs(self):
        return [epoch_result.avg_packet_acc for epoch_result in self.epoch_results]

    @property
    def avg_loss(self):
        if len(self) > 0:
            return np.mean(self.epoch_avg_losses)
        else:
            return float("inf")

    @property
    def avg_acc(self):
        if len(self) > 0:
            return np.mean(self.epoch_avg_accs)
        else:
            return float("inf")

    # Generics #
    def __len__(self):
        return len(self.epoch_results)


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


def process_model_metrics(metrics: Dict[str, ModelMetrics]):
    """
    @Description: Plots and prints an overview of the model training along with metrics

    @Notes:
        - Plot the epoch training and validation curves on the same graph

    @Returns:
    """
    train_metrics = metrics["train"]
    val_metrics = metrics["validation"]

    # Create a plot of the average epoch losses for each
    plt.plot(train_metrics.epoch_avg_losses)
    plt.plot(val_metrics.epoch_avg_losses)
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    title = f"Overall training and validation loss"
    plt.title(title)
    plt.show()

    # Accuracy plot
    plt.plot(train_metrics.epoch_avg_accs)
    plt.plot(val_metrics.epoch_avg_accs)
    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy")
    title = f"Overall training and validation Accuracy"
    plt.title(title)
    plt.show()

    # Conversation length vs accuracy
    conv_lens = [
        str(epoch_result.max_conv_len) for epoch_result in train_metrics.epoch_results
    ]
    plt.plot(train_metrics.epoch_avg_accs)
    plt.plot(val_metrics.epoch_avg_accs)
    plt.xticks(np.arange(len(conv_lens)), conv_lens)
    plt.xlabel("Packets per conversation")
    plt.ylabel("Accuracy")
    title = f"Overall Accuracy vs Conversation length"
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


def create_micro_conversations(
    packet_loader: PacketIterator,
    conv_len_rng: Tuple[int, int] = (MIN_MICRO_CONV_LEN, MAX_MICRO_CONV_LEN),
) -> Iterator[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """
    @Description: Splits a packet iterator into many different sub conversations

    @Notes:
        - A generic packet loader draws no distinctions between conversation boundaries.

    @Returns: Iterator of PacketIterator each representing a micro conversation (chatter)
    """
    assert (
        conv_len_rng[0] <= conv_len_rng[1]
    ), f"The conversation length range lb must be less than ub {conv_len_rng}"
    last_conv_num = None
    micro_conv = list()

    for packet in packet_loader:
        if len(packet[2]) == 0:
            continue

        cur_conv_num = packet[0][3]  # Conversation number

        if last_conv_num != cur_conv_num:
            if len(micro_conv) >= conv_len_rng[0]:
                yield micro_conv

            micro_conv = list()

        elif len(micro_conv) >= conv_len_rng[1]:
            yield micro_conv
            micro_conv = list()

        elif len(micro_conv) >= conv_len_rng[0]:
            if random.random() < MICRO_CONV_YIELD_PROB:
                yield micro_conv
                micro_conv = list()

        last_conv_num = cur_conv_num

        micro_conv.append(packet)


class PacketItGenerator:
    def __init__(self, csv_dir: str):
        self.csv_dir = csv_dir
        self.n_conv_packets = N_MAX_CONV_PACKETS
        self.cur_packet_n = 0
        self.conv_list = list()
        self.split_dict = {"train": [], "val": [], "test": []}
        self.dfs = load_dfs_from_dir(csv_dir=csv_dir)

    ### Private ###
    @staticmethod
    def split_into_conversations(
        df: pd.DataFrame,
        conv_list: List[Tuple[str, str]],
        add_conv_num: bool = True,
    ) -> List[pd.DataFrame]:
        """
        @Description: Takes a data frame from the mqtt data and divides it into separate
        conversations each indexed by a uniqe conversation number. The conv_list is
        provided to ensure that conversation number uniqueness remains consistent throughout
        multiplie conversation splits. If starting fresh just pass an empty list.

        @Notes:
            - The conv_list is modified if there a new conversation is found. Appending to a list
            passed to a function modifies that list and not just a copy of it.

        """
        ips = list(col_values_set(df, SRC_IP_TAG).keys())

        convs = list()

        while len(ips) > 1:
            ip1 = ips.pop()

            for ip2 in ips:
                conv_df = conversation_filter(df, ip1, ip2).copy()
                if len(conv_df) > 0:
                    # Search for the conversation in the conv list
                    conv_num = len(conv_list)
                    for i, (c_ip1, c_ip2) in enumerate(conv_list):
                        if ((c_ip1 == ip1) and (c_ip2 == ip2)) or (
                            (c_ip1 == ip2) and (c_ip2 == ip1)
                        ):
                            conv_num = i
                            break

                    if conv_num == len(conv_list):
                        conv_list.append((ip1, ip2))

                    if add_conv_num:
                        conv_df["conv.number"] = [conv_num] * len(conv_df)

                    convs.append(conv_df)

        return convs

    def update_split_dict(self):
        n_convs = len(self.conv_list)
        n_in_dict = np.sum([len(elem) for elem in self.split_dict.values()])
        # Number of values to add for each category
        n_train = int(TRAIN_VAL_TEST_PERCS[0] * n_convs) - len(self.split_dict["train"])
        n_val = int(TRAIN_VAL_TEST_PERCS[1] * n_convs) - len(self.split_dict["val"])
        n_test = n_convs - n_val - n_train - len(self.split_dict["test"])

        # Now get how many are already in the split dict
        new_conv_nums = list(range(n_in_dict, n_in_dict + n_train + n_val + n_test))
        for key, n_nums in zip(self.split_dict.keys(), (n_train, n_val, n_test)):
            for _ in range(n_nums):
                # We randomly choose which category each conversation nubmer should go to
                conv_num = new_conv_nums.pop(np.random.randint(0, len(new_conv_nums)))
                self.split_dict[key].append(conv_num)

        assert (
            len(new_conv_nums) == 0
        ), f"Remaining converstion number list length must be 0, not {len(new_conv_nums)}"

    def split_convs(
        self, conv_dfs: List[PacketDataset]
    ) -> Dict[str, List[PacketDataset]]:
        # update the splti dict
        self.update_split_dict()

        ret = {"train": [], "val": [], "test": []}
        # Now use the indicies to split the conversations into train, validation, and test
        # We assume that each conv_df has one and only one conversation number
        for conv_df in conv_dfs:
            for key, conv_nums in self.split_dict.items():
                if conv_df.conv_num in conv_nums:
                    ret[key].append(conv_df)

        return ret

    def packet_it_generator(
        self, df_split: List[PacketDataset]
    ) -> Iterator[ParsedPacket]:
        """
        @Description: Creates a batch stream of parsed packets from the given conversations

        @Notes:
            - We return if the end of a conversation is reached before the full batch lenght is reached

        @Returns: (cat_features_tensor, numerical_features_tensor, payloads_tensor)
        """
        # Use the epoch number to schedule how many packets will we will select from each conversation
        for df in df_split:
            for i, packet in enumerate(df):
                if i > self.n_conv_packets:
                    break
                self.cur_packet_n = i
                yield packet

    ### Public ###
    def update_n_packets(self, epoch_num: int = N_NUM_EPOCHS - 1):
        self.n_conv_packets = int(N_MAX_CONV_PACKETS * (1 + epoch_num) / N_NUM_EPOCHS)

    def generate_loaders(self, epoch_num: int | None = None) -> Tuple[
        PacketIterator,
        PacketIterator,
        PacketIterator,
    ]:
        """
        @Description: Takes a general dataset, splits it into validation and training and then
        creates loaders for each data split

        @Notes:

        @Returns:
        """
        train_dfs = list()
        validation_dfs = list()
        test_dfs = list()

        if epoch_num is not None:
            self.update_n_packets(epoch_num=epoch_num)

        train_len = 0
        validation_len = 0
        test_len = 0

        cat_dims = list()
        num_dims = 0
        # Load the dataset
        for df in self.dfs:
            # Get the conversations splits
            splits = self.split_into_conversations(df, conv_list=self.conv_list)

            conv_dfs = [
                PacketDataset(conv_df, n_convs=len(self.conv_list))
                for conv_df in splits
            ]

            train, validation, test = self.split_convs(conv_dfs).values()

            cat_dims = train[0].cat_dims
            num_dims = train[0].num_dims

            train_dfs += train
            validation_dfs += validation
            test_dfs += test

            for t in train:
                train_len += len(t)

            for v in validation:
                validation_len += len(v)

            for ts in test:
                test_len += len(ts)

        assert isinstance(cat_dims, list), f"The cat dims must be a list of integers"
        assert isinstance(num_dims, int), f"The num dims must be an integer"

        # Now create batch generators for each
        return (
            PacketIterator(
                self.packet_it_generator(train_dfs),
                train_len,
                cat_dims=cat_dims,
                num_dims=num_dims,
            ),
            PacketIterator(
                self.packet_it_generator(validation_dfs),
                validation_len,
                cat_dims=cat_dims,
                num_dims=num_dims,
            ),
            PacketIterator(
                self.packet_it_generator(test_dfs),
                test_len,
                cat_dims=cat_dims,
                num_dims=num_dims,
            ),
        )

    def generate_conv_loaders(self) -> Dict[str, List[PacketDataset]]:
        # Load the csv file data frames from the directory
        train = list()
        validation = list()
        testing = list()

        for csv_df in self.dfs:
            splits = self.split_into_conversations(csv_df, self.conv_list)

            conv_dfs = [
                PacketDataset(conv_df, n_convs=len(self.conv_list))
                for conv_df in splits
            ]

            # Split the conv dfs into training, validation, and testing
            s_train, s_validation, s_testing = self.split_convs(
                conv_dfs=conv_dfs
            ).values()

            train += s_train
            validation += s_validation
            testing += s_testing

        return {"train": train, "validation": validation}


### File system helpers ###
def pkl_write_model(
    model,
    metrics: Dict[str, ModelMetrics],
    save_dir: str,
    metadata: Dict = {},
):
    """
    @Description: Takes the model metrics generated during the training process and saves them to
    pickle file so that they can be read back again for analysis.

    @Notes:
        - Any other relevant or interesting info can be passed through the metadata dictionary

    @Returns:
    """
    assert os.path.isdir(save_dir), f"E: The directory {save_dir} does not exist"

    # Save the model
    model_path = os.path.join(save_dir, f"model_{type(model).__name__}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump((model, metadata), f)

    # Save the metrics
    metrics_path = os.path.join(save_dir, O_RESULTS_FNAME)
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)

    # Save the metadata
    metadata_path = os.path.join(save_dir, O_METADATA_FNAME)
    metadata["git_hash"] = get_git_hash() # Indication of 
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)


def pkl_read_model(read_dir) -> Tuple[Any, Dict[str, ModelMetrics], Dict]:
    assert os.path.isdir(read_dir), f"E: The directory {read_dir} does not exist"

    model_path = None
    for path in os.listdir(read_dir):
        if "model_" in path:
            model_path = os.path.join(read_dir, path)
            break

    model = None
    if model_path is not None:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    metrics_path = os.path.join(read_dir, O_RESULTS_FNAME)
    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)

    metadata_path = os.path.join(read_dir, O_METADATA_FNAME)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return model, metrics, metadata


def get_git_hash() -> str:
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        return git_hash
    except Exception as e:
        print(f"E: Get git has failed with {e}")
        return ""


### Custom loss functions ###
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = P_SMOOTHING):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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


def sequence_memorization_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    memory_bank: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """
    @Description: Takes the dictionary of the past n most frequent sequences, checks if the current
    prediction is identical and invokes a punishment if that is the case.

    @Notes:
        - Currently in this loss function there is no reward for getting the entire sequence correct
        Since we are using an ensamble of loss functions, the others will yield reward for a correct
        predition. This is simply intended to generalize

    @Returns:
    """
    if memory_bank is None:
        return torch.tensor(0.0, device=predictions.device)

    # Convert predictions to string key for lookup
    pred_seq = tuple(predictions.cpu().numpy())
    target_seq = tuple(targets.cpu().numpy())

    # Penalize if this exact sequence has been seen many times
    seq_key = str(pred_seq)
    frequency = memory_bank.get(seq_key, 0)

    # Higher penalty for more frequent sequences
    if frequency > 2:  # If seen more than 2 times
        penalty = torch.tensor(float(frequency - 2) * 0.1, device=predictions.device)
        return penalty

    return torch.tensor(0.0, device=predictions.device)


def apply_sequence_augmentation(
    payload: torch.Tensor, epoch: int, total_epochs: int
) -> torch.Tensor:
    """
    @Description:

    @Notes:

    @Returns:
    """

    # If the sequence is too short, don't purturb it any more
    if len(payload) <= R_MIN_KEEP_LENGTH:
        return payload

    augmented = payload.clone()

    # Progressive augmentation - more aggressive early in training
    aug_intensity = max(0.1, 1.0 - (epoch / total_epochs))

    # 1. Random byte dropping this actually shortens the sequence
    # TBD! Since the sequence length is something provided in the meta data, we would like to
    # keep the lengths the same. Masking provides a similar result without the nasty side effects
    # if random.random() < R_SEQUENCE_DROP_PROB * aug_intensity:
    #     max_drop = min(
    #         int(len(payload) * R_MAX_DROP_RATIO), len(payload) - R_MIN_KEEP_LENGTH
    #     )
    #     if max_drop > 0:
    #         n_drop = random.randint(1, max_drop)
    #         drop_indices = random.sample(range(len(payload)), n_drop)
    #         # Create mask and remove dropped indices
    #         mask = torch.ones(len(payload), dtype=torch.bool)
    #         mask[drop_indices] = False
    #         augmented = augmented[mask]

    # 2. Random subsequence shuffling
    if random.random() < R_SEQUENCE_SHUFFLE_PROB * aug_intensity and len(augmented) > 4:
        # Shuffle small chunks to break local patterns
        chunk_size = random.randint(2, min(4, len(augmented) // 2))
        start_idx = random.randint(0, len(augmented) - chunk_size)
        chunk = augmented[start_idx : start_idx + chunk_size].clone()
        # Shuffle within chunk
        shuffle_idx = torch.randperm(chunk_size)
        augmented[start_idx : start_idx + chunk_size] = chunk[shuffle_idx]

    # 3. Random masking of bytes
    if random.random() < R_SEQUENCE_MASK_PROB * aug_intensity:
        n_mask = random.randint(1, min(3, len(augmented)))
        mask_indices = random.sample(range(len(augmented)), n_mask)
        for idx in mask_indices:
            augmented[idx] = MASK

    return augmented


def progressive_loss(loss_vector: torch.Tensor) -> torch.Tensor:
    """
    @Description: Generates a loss value from how our loss vector progresses.
    Ideally we would like to see the conversation loss begin high because the memory is not yet
    established and undergo exponential decay as we add packets

    @Notes:
        - L = max_l + (start_l - end_l) * dx
        - All operations preserve gradients
        - Only max, first, and last losses get gradients

    @Returns: Scalar loss with gradients connected to model
    """
    if len(loss_vector) == 0:
        return torch.tensor(0.0, device=loss_vector.device, requires_grad=True)

    if len(loss_vector) == 1:
        return loss_vector[0]

    # All these operations preserve gradients
    max_loss = loss_vector.max()
    improvement = loss_vector[0] - loss_vector[-1]
    sequence_length = float(
        loss_vector.numel()
    )  # Convert to float, no gradients needed

    return torch.sigmoid((max_loss + improvement * sequence_length) / 10) * 10


def conversation_tradjectory_loss(loss_vector: torch.Tensor) -> torch.Tensor:
    """
    Alternative: Focus on the entire tradjectory shape
    This gives gradients to ALL elements in the loss vector
    """
    if len(loss_vector) <= 1:
        return (
            loss_vector[0]
            if len(loss_vector) == 1
            else torch.tensor(0.0, device=loss_vector.device)
        )

    # Ideal tradjectory: exponential decay
    # Create target tradjectory
    n_steps = len(loss_vector)
    x = torch.arange(n_steps, dtype=torch.float32, device=loss_vector.device)

    # Target: start high, decay exponentially
    initial_loss = loss_vector[0].detach()  # Use actual first loss as target start
    target_decay = initial_loss * torch.exp(-x * 0.5)  # Exponential decay

    # MSE between actual tradjectory and ideal tradjectory
    tradjectory_loss = F.mse_loss(loss_vector, target_decay)

    # Also add improvement term
    improvement = loss_vector[0] - loss_vector[-1]

    return tradjectory_loss - 0.1 * improvement  # Reward improvement


def conversation_trajectory_loss_simple(losses: torch.Tensor):
    """Simple trajectory loss - reward improvement over conversation"""

    # Reward decreasing loss over time
    if len(losses) > 1:
        # Compute improvement: early_loss - late_loss (positive = improvement)
        early_avg = losses[: len(losses) // 2].mean()
        late_avg = losses[len(losses) // 2 :].mean()
        improvement = early_avg - late_avg

        # Loss should be negative of improvement (we want to maximize improvement)
        return -improvement + losses.mean()  # Base loss + improvement penalty
    else:
        return losses.mean()
