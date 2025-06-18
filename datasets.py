"""
@Author: Joel Weber
@Date: 2025-06-09
@Description: At this point all the data has been preprocessed and is ready
to be learned from. This includes various predictive model implementations
along with some huristics to test and score the models

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
from icecream import ic

# Local imports
from CONSTANTS import *
from preprocessing import extract_features, split_into_conversations
from datasets import PacketDataset


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
    """

    def __init__(self, df: pd.DataFrame, seq_len: int = 128):
        self.features = extract_features(df)
        self.seq_len = seq_len
        self.samples = self._create_samples()
        self.cnt = 0

        self._create_samples()

    def _create_samples(self) -> Iterator:
        """
        @Description: This creates inputs for each next byte. It uses both the
        categorical / numerical features tied to each frame along with the context
        of the last bytes.

        @Notes:
            - This includes a sequence length which is an attention mask that will use
            the inputs of the past to determine the next.
            - The features are give as a dictionary where the class of values and actual list
            of data are both given.
            - At the momenent we only support one sequence feature
            - Regardless of how many characters we have, we always begin predicting at the
            first byte with a zero sequence length and then building a context up from there.
            - We use padding and masking for short sequence lengths
        @Returns:
        """
        cat_features = list()
        num_features = list()
        seq_features = list()

        self.cat_dims = 0
        self.num_dim = 0
        self.seq_dim = 0

        for name, data in self.features.items():
            dtype = data["dtype"]
            values = data["values"]
            if dtype == "categorical":
                cat_features.append(values)
                self.cat_dims += data["dim"]
            elif dtype == "numerical":
                num_features.append(values)
                self.num_dim += data["dim"]
            elif dtype == "sequential":
                seq_features.append(values)
                self.seq_dim += data["dim"]
            else:
                ic(
                    f"{name} has unrecognized dtype: {dtype} given for feature, ignoring ..."
                )

        assert (
            len(seq_features) == 1
        ), f"Currently one and only one sequential feature is permitted, not {len(seq_features)}"
        # Now go through all the features and add them to embedding layer
        for i, (cat_f, num_f, seq_f) in enumerate(
            zip(zip(*cat_features), zip(*num_features), zip(*seq_features))
        ):

            seq = seq_f[0]
            for i in range(len(seq) - 1):
                # Create the window. Please not that initially we begin with no context besides
                # The history, packet meta data, and model parameters. This is intentional.
                input_len = min(i, self.seq_len)
                input_bytes = list(seq[i - input_len : i]) + [0] * (
                    self.seq_len - input_len
                )
                target_byte = seq[i]
                mask = [1] * input_len + [0] * (self.seq_len - input_len)

                # The categorical and numerical features are constant throughout the packet
                self.cnt += 1
                yield {
                    "input_bytes": torch.tensor(input_bytes, dtype=torch.long),
                    "attention_mask": torch.tensor(mask, dtype=torch.bool),
                    "seq_len": input_len,
                    "position": torch.tensor(i, dtype=torch.long),
                    "input_numerical": torch.tensor(num_f, dtype=torch.long),
                    "input_categorical": torch.tensor(cat_f, dtype=torch.long),
                    "target": torch.tensor(target_byte, dtype=torch.long),
                }

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.samples)
        except StopIteration:
            ic(f"Samples end reached with total count of {self.cnt}")
            raise StopIteration


if __name__ == "__main__":
    from preprocessing import load_df

    # Get the packet data set
    df = load_df()

    conv_dfs = split_into_conversations(df)

    def test_packet_dataset(df: pd.DataFrame):
        ds = PacketDataset(df)

        i = 0
        while i < 1000:
            sample = next(ds)
            # print(sample)
            i += 1

    for conv_df in conv_dfs:
        print(len(conv_df))
        test_packet_dataset(conv_df)
