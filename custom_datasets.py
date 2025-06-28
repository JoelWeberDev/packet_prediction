"""
@Author: Joel Weber
@Date: 2025-06-09
@Description: At this point all the data has been preprocessed and is ready
to be learned from. This includes various predictive model implementations
along with some huristics to test and score the models

Each of the packets needs to be divided into packets with a contextual history.
For each trainging point we need the history, packet metadata, and the payload with which
to compare it to.

@Notes:
    - Here is a list of what features


@TODO:

"""

# Library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Optional, Iterator
from dataclasses import dataclass
from icecream import ic

# Local imports
from CONSTANTS import *
from preprocessing import extract_features, split_into_conversations


### Custom data structures ###


@dataclass
class ParsedPacket:
    padded_payload: torch.Tensor
    cat_names: list
    cat_features: torch.Tensor
    numerical_names: list
    numerical_features: torch.Tensor

    def __str__(self):

        ret = f"cat_features: {[f'{name}: {value}' for name, value in zip(self.cat_names, self.cat_features.tolist())]}\n"
        ret += f"numerical_features: {[f'{name}: {value}' for name, value in zip(self.numerical_names, self.numerical_features.tolist())]}\n"
        ret += f"{self.padded_payload}\n"

        return ret


@dataclass
class PacketWithContext:
    context: List[ParsedPacket]
    target: ParsedPacket


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


### Byte sequence dataset ###
class PacketDataset(Dataset):
    """
    @Description: Dataset of parsed network packets with all the relevant categorical features
    and the byte sequences from the packet payload. This will take a pandas dataframe, do
    some preprocessing and then convert into the Dataset format for tensor flow.

    @Notes:
        - In the dataset there are numerical, sequential, and categorical features. All of these
        must be married into the same framework
        - This returns an iterator that so that the results can be used on demand rather than being
        stored all in memory.
        Categoical feature structure follows the format:
            [mqtt.hdrcmd, mqtt.hdrflags, flow.direction, conv.number]
        Numerical feature structure:
            [frame.number, frame.time_delta, mqtt.len]
    """

    def __init__(self, df: pd.DataFrame, n_convs: int, seq_len: int = MAX_SEQ_LEN):
        self.features = extract_features(df, n_convs=n_convs)
        self.seq_len = seq_len
        self.cnt = 0
        self.df_len = len(df)
        self.history = list()

        self._process_packets()

    def _process_packets(self):
        """
        @Description: Prepares the dimensionality of the dataset

        @Notes:

        @Returns:
        """
        # Divide the features into categorical, numerical, and payload
        self.cat_features = dict()
        self.num_features = dict()
        self.seq_features = dict()

        self.cat_dims = list()
        self.num_dim = 0
        self.seq_dim = 0

        for name, data in self.features.items():
            dtype = data["dtype"]
            values = data["values"]
            if isinstance(values, np.ndarray):
                values = values.tolist()
            if dtype == "categorical":
                self.cat_features[name] = values
                self.cat_dims.append(data["dims"])
            elif dtype == "numerical":
                self.num_features[name] = values
                self.num_dim += data["dims"]
            elif dtype == "sequential":
                self.seq_features[name] = values
                self.seq_dim += data["dims"]
            else:
                ic(
                    f"{name} has unrecognized dtype: {dtype} given for feature, ignoring ..."
                )

        print(
            f"cat_dims: {self.cat_dims}, num_dims: {self.num_dim}, seq_dims: {self.seq_dim}"
        )

        assert (
            len(self.seq_features) == 1
        ), f"Currently one and only one sequential feature is permitted, not {len(self.seq_features)}"

        self.packets = self._parse_packets()

    def _parse_packets(self) -> Iterator[PacketWithContext]:
        """
        @Description: Parses the data frame into a packets with context that are suitable for
        training and inference

        @Notes:

        @Returns:
        """
        # Now run through the data
        for i, (cat_f, num_f, seq_f) in enumerate(
            zip(
                zip(*self.cat_features.values()),
                zip(*self.num_features.values()),
                zip(*self.seq_features.values()),
            )
        ):
            self.cnt += 1
            # Create a Sequence input from the parsed features
            seq = seq_f[0]
            seq_len = len(seq)
            # Now pad the sequence
            if seq_len > MAX_SEQ_LEN:
                # We simply lose data here
                seq = seq[:MAX_SEQ_LEN]
                seq_len = MAX_SEQ_LEN

            elif len(seq) < MAX_SEQ_LEN:
                seq = torch.tensor(
                    seq.tolist() + [0] * (MAX_SEQ_LEN - seq_len), dtype=torch.long
                )

            attn_mask = torch.tensor(
                ([1] * seq_len) + ([0] * (MAX_SEQ_LEN - seq_len)), dtype=torch.bool
            )

            assert (
                len(seq) == MAX_SEQ_LEN
            ), f"The sequence length {len(seq)} must equal {MAX_SEQ_LEN}"

            assert (
                len(attn_mask) == MAX_SEQ_LEN
            ), f"The sequence length {len(attn_mask)} must equal {MAX_SEQ_LEN}"

            pop = ParsedPacket(
                torch.tensor(seq),
                list(self.cat_features.keys()),
                torch.tensor(list(cat_f), dtype=torch.long),
                list(self.num_features.keys()),
                torch.tensor(list(num_f), dtype=torch.long),
            )

            if i < CONV_CONTEXT_LEN:
                self.history.append(pop)
                continue

            # slice for most recent history
            ret = PacketWithContext(self.history.copy(), pop)
            self.history.append(pop)
            self.history = self.history[-CONV_CONTEXT_LEN:]

            yield ret

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.packets)
        except StopIteration:
            ic(f"Batches end reached with total count of {self.cnt}")
            raise StopIteration

    def __len__(self):
        return self.len


if __name__ == "__main__":
    from preprocessing import load_df

    # Get the packet data set
    df = load_df()

    conv_dfs = split_into_conversations(df)

    def test_packet_dataset(df: pd.DataFrame):
        ds = PacketDataset(df, 1)

        pkg = next(ds)

        print(pkg.target)

    for conv_df in conv_dfs:
        print(len(conv_df))
        test_packet_dataset(conv_df)
