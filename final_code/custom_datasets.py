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
import torch
from torch.utils.data import Dataset
from typing import List, Iterator
from dataclasses import dataclass
from icecream import ic

# Local imports
from final_code.CONSTANTS import *
from final_code.preprocessing import extract_features


### Custom data structures ###
@dataclass
class ParsedPacket:
    payload: List[int]
    cat_names: list
    cat_features: torch.Tensor
    numerical_names: list
    numerical_features: torch.Tensor

    ### Generics ###
    def __str__(self):

        ret = f"cat_features: {[f'{name}: {value}' for name, value in zip(self.cat_names, self.cat_features.tolist())]}\n"
        ret += f"numerical_features: {[f'{name}: {value}' for name, value in zip(self.numerical_names, self.numerical_features.tolist())]}\n"
        ret += f"{self.payload}\n"

        return ret


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

    def __init__(self, df: pd.DataFrame, n_convs: int):
        self.features = extract_features(df, n_convs=n_convs)
        self.cnt = 0
        self.len = len(df)
        self.history = list()
        number_list = self.features["conv.number"]["values"]
        self.conv_num = int(number_list[0])

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
        self.num_dims = 0
        self.seq_dims = 0

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
                self.num_dims += data["dims"]
            elif dtype == "sequential":
                self.seq_features[name] = values
                self.seq_dims += data["dims"]
            else:
                ic(
                    f"{name} has unrecognized dtype: {dtype} given for feature, ignoring ..."
                )

        print(
            f"cat_dims: {self.cat_dims}, num_dims: {self.num_dims}, seq_dims: {self.seq_dims}"
        )

        assert (
            len(self.seq_features) == 1
        ), f"Currently one and only one sequential feature is permitted, not {len(self.seq_features)}"

        self.packets = self._parse_packets()

    def _parse_packets(self) -> Iterator[ParsedPacket]:
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

            pop = ParsedPacket(
                seq_f[0].tolist(),
                list(self.cat_features.keys()),
                torch.tensor(list(cat_f), dtype=torch.long),
                list(self.num_features.keys()),
                torch.tensor(list(num_f), dtype=torch.long),
            )

            yield pop

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
