"""
@Author: Joel Weber
@Date: 2025-07-25
@Description: This is a completely different concept that chooses to address the overfitting issue
by embracing it with online learning. The architechure is setup to have support components that will
be frozen at runtime and then an intenal model such as a GRU, LSTM, or ESN actively being updated
as the conversation progresses. 

@Notes: 
- The fundamental issue will be, how do we train the support components to rely on the internal 
micro model rather than themselves memorizing the training data.
    - My proposed solution is to train the support architechure at the conversation level and 
    the micro model at the packet level.

@TODO: 
"""
### Python imports ###
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Iterator
import random

### Local imports ###
from modules.CONSTANTS import *
from modules.preprocessing import load_df, load_dfs_from_dir, split_into_conversations
from modules.custom_datasets import PacketDataset
from modules.helper_functions import (
    print_update,
    plot_metrics,
    hidden_reliance_loss,
    compute_hidden_state_regularization,
    LabelSmoothingCrossEntropy,
    PacketItGenerator,
)


class OnlinePacketPredictor(nn.Module):
    """
    @Description: Model implementing online learning at the conversation level for predicting 
    the next packet.
    
    @Notes: 
    
    """

    def __init__(
        self,
        categorical_dims: List[int],
        numerical_dim: int,
        hidden_size: int = N_HIDDEN_SIZE,
        embedding_size: int = N_EMB_SIZE,
        num_layers: int = N_NUM_LAYERS,
        dropout: float = N_DROPOUT,
    ):
        super().__init__()

        # 1. Metadata Encoding
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(dim, min(embedding_size, (dim + 1) // 2))
                for dim in categorical_dims
            ]
        )

        cat_embed_dim = sum(
            min(embedding_size, (dim + 1) // 2) for dim in categorical_dims
        )
        metadata_dim = cat_embed_dim + numerical_dim

        # Metadata MLP with normalization
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )  # Inference Frozen

        # 2. Byte Embeddings
        self.byte_embedding = nn.Embedding(
            VOCAB_DIM, embedding_size
        )  # Inference Frozen

        


