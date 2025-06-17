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
from typing import Dict, Tuple, List, Optional
from icecream import ic

# Local imports
from CONSTANTS import *
from preprocessing import extract_features


### Byte sequence dataset ###
class PacketDataset(Dataset):
    """
    @Description: Dataset of parsed network packets with all the relevant categorical features
    and the byte sequences from the packet payload. This will take a pandas dataframe, do
    some preprocessing and then convert into the Dataset format for tensor flow.

    @Notes:
        - In the dataset there are numerical, sequential, and categorical features. All of these
        must be married into the same framework
    """

    def __init__(self, df: pd.DataFrame, seq_len: int = 128):
        self.features = extract_features(df)
        self.seq_len = seq_len
        self.samples = list()

    def _create_samples(self):
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
        @Returns:
        """
        cat_features = list()
        num_features = list()
        seq_features = list()

        for name, data in self.features.items():
            for dtype, values in data.values():
                if dtype == "categorical":
                    cat_features.append(values)
                elif dtype == "numerical":
                    num_features.append(values)
                elif dtype == "sequential":
                    seq_features.append(values)
                else:
                    ic(
                        f"{name} has unregocnized dtype: {dtype} given for feature, ignoring ..."
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
                input_bytes = seq[max(0, i - self.seq_len) : i]
                target_byte = seq[i]

                # The categorical and numerical features are constant throughout the packet
                self.samples.append(
                    {
                        "input_byte": torch.tensor(input_bytes, dtype=torch.long),
                        "input_numerical": torch.tensor(num_f, dtype=torch.long),
                        "input_categorical": torch.tensor(cat_f, dtype=torch.long),
                        "target": torch.tensor(target_byte, dtype=torch.long),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
