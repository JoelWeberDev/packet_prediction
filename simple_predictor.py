"""
@Author: Joel Weber
@Date: 2025-06-28
@Description: This is a simple lstm model that uses a fixed conversation byte context size
along with the packet meta data to predict the next byte in the sequence. To generate an
entire next packet we auto regressivly predict the next byte until the next start of sentence
SOS token is reached.

@Notes:
    - All the features are repeated for the each byte. This does lead to a slight increase in
    model size, but saves on simplicity.
    - The cell and hidden states must be explicitly preserved throughout calls.

@TODO:
    - Create embedding for the categorical and numerical features
    - Construct a context embedder
    - Make an autoregressive training process that batches the sequences up into fixed lengths
    and preserves cell and hidden states for each batch.

"""

### Python imports ###
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import List, Dict, Tuple

### Local imports ###
from preprocessing import load_df, split_into_conversations
from custom_datasets import PacketDataset, ConversationByteStream, Byte, ByteWithContext
from CONSTANTS import *


# Next byte predictor
class NextBytePredictor(nn.Module):
    """
    @Description: The primary lstm model that handles embeddings, prediction, state preservation,
    and output processing

    @Notes:
        - The past bytes, meta data, and current packet bytes are all embedded equally. By design
        the lstm model will learn how to weight each input embedding to best predict the next byte
        - All the meta data is repeated for each byte. Although this is a waste we save on simplicity
        and need for something like an alternate mlp to embed the meta data separately
        - We train using the true context rather than the generated context because a single error
        in the generation can cause tragic results for the rest of the sequence. Much initial training
        data would be wasted if we train in this way.
        - Due to the nature of autoregressive systems the generation will suffer from any errors.
        Therefore we need to prioritize getting bytes exactly correct, not just almost correct.
        - The total embedding dims is [BYTE_EMBED_DIM * S_BYTE_CTX_LEN + cat_embed_dims + num_embed_dims]
    """

    def __init__(self, cat_dims: List[int], numerical_dims: int, device: str = DEVICE):
        super().__init__()

        # Used to embed a single byte or generate a tensor of embeddings for a tensor of bytes
        self.byte_embedding = nn.Embedding(INPUT_VOCAB_DIM, BYTE_EMBED_DIM)

        def get_embedding_dim(n_cats: int) -> int:
            # Google's categorical embedding formuala
            return min(MAX_CAT_EMB, round((n_cats * CAT_EMB_SCALAR) ** CAT_EMB_EXPO))

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(cat_size, get_embedding_dim(cat_size))
                for cat_size in cat_dims
            ]
        )

        self.cat_embed_dim = sum(get_embedding_dim(cat_size) for cat_size in cat_dims)

        # Since all the numerical features are already longs we can simply give them on dimension
        self.numerical_embed_dim = numerical_dims

        self.input_size = S_BYTE_CTX_LEN * (
            BYTE_EMBED_DIM + self.cat_embed_dim + self.numerical_embed_dim
        )

        # Now create the acutal LSTM for inference
        self.next_byte_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=S_HIDDEN_SIZE,
            num_layers=S_LSTM_LAYERS,
            batch_first=True,
            dropout=S_LSTM_DROPOUT,
        )

        # Now create the output projector
        self.output_projection = nn.Linear(S_HIDDEN_SIZE, OUTPUT_VOCAB_DIM)

        self.device = device
        self.to(device=device)

    def forward(
        self,
        input_batch: List[ByteWithContext],
        hidden_states: torch.Tensor | None = None,
        cell_states: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        @Description: Here we take a batch of BytesWithContext and generate the next byte in
        the sequence for each of those.

        @Notes:

        @Returns: Output logits torch.Tensor([batch_size, OUTPUT_VOCAB_SIZE])
        """

        assert all(
            [isinstance(inp, ByteWithContext) for inp in input_batch]
        ), f"All the inputs in the batch must be ByteWithContext objects"
        batch_size = len(input_batch)

        # Create the embeddings for each of the byte context objects
        embedded_inputs = torch.stack(
            [self._embed_byte_context(byte_with_ctx) for byte_with_ctx in input_batch],
            dim=1,
        ).to(self.device)

        # Now run the next byte prediction lstm
        hidden_states = (
            torch.zeros((batch_size, S_HIDDEN_SIZE), dtype=torch.long)
            if hidden_states is None
            else hidden_states
        )
        assert hidden_states.shape == (
            batch_size,
            S_HIDDEN_SIZE,
        ), f"The hidden states must have shape {(batch_size, S_HIDDEN_SIZE)}, not {hidden_states.shape}"

        cell_states = (
            torch.zeros((batch_size, S_HIDDEN_SIZE), dtype=torch.long)
            if cell_states is None
            else cell_states
        )
        assert cell_states.shape == (
            batch_size,
            S_HIDDEN_SIZE,
        ), f"The cell states must have shape {(batch_size, S_HIDDEN_SIZE)}, not {cell_states.shape}"

        byte_outputs, (hidden_states, cell_states) = self.next_byte_lstm(
            embedded_inputs, (hidden_states, cell_states)
        )

        assert isinstance(hidden_states, torch.Tensor) and isinstance(
            cell_states, torch.Tensor
        ), f"The hidden and cell states must be tensors not {type(hidden_states)}, and {type(cell_states)}"

        logits = self.output_projection(byte_outputs)

        return logits, hidden_states, cell_states

    ### Helper functions ###
    def _embed_byte(self, byte: Byte) -> torch.Tensor:
        """
        @Description: Completely embeds a single byte object that contains a byte with
        categorical and numerical meta data features into a properly sized tensor

        @Returns: torch.Tensor([BYTE_EMBED_DIM + cat_embed_dim + num_embed_dim])
        """
        cat_emb_list = list()
        for i, cat_emb_layer in enumerate(self.cat_embeddings):
            cat_emb_list.append(cat_emb_layer(byte.cat_features[i]))
        cat_embs = (
            torch.cat(cat_emb_list, dim=-1)
            if cat_emb_list
            else torch.empty(0, dtype=torch.long)
        )

        byte_emb = self.byte_embedding(byte.value)

        return torch.cat([byte_emb, cat_embs, byte.numerical_features], dim=-1)

    def _embed_byte_context(self, byte_with_ctx: ByteWithContext) -> torch.Tensor:
        """
        @Description: Completely embeds the byte with context object into the
        properly sized tensor

        @Notes:
            - This only embeds the context. The target byte does not need to be embedded

        @Returns: torch.Tensor([S_BYTE_CTX_LEN * (BYTE_EMBED_DIM + cat_embed_dim + num_embed_dim)])
        """
        return torch.cat(
            [self._embed_byte(byte) for byte in byte_with_ctx.context], dim=-1
        )


### Training and validation functions ###
def split_convs(conv_dfs: List[PacketDataset]) -> Dict[str, List[PacketDataset]]:
    train_idx = int(TRAIN_VAL_TEST_PERCS[0] * len(conv_dfs))
    val_idx = int(TRAIN_VAL_TEST_PERCS[1] * len(conv_dfs))

    assert (
        train_idx > 0
    ), f"Insufficient training convs: The number of conversations {len(conv_dfs)} * {TRAIN_VAL_TEST_PERCS[0]} is below 1, please provide more conversations"
    assert (
        val_idx > 0
    ), f"Insufficient validations convs: The number of conversations {len(conv_dfs)} * {TRAIN_VAL_TEST_PERCS[1]} is below 1, please provide more conversations"

    return {
        "train": conv_dfs[:train_idx],
        "val": conv_dfs[train_idx : train_idx + val_idx],
        "test": conv_dfs[train_idx + val_idx :],
    }


def model_train():
    """
    @Description: Creates a fresh model, and trains it on the dataset

    @Notes:

    @Returns:
    """
    df = load_df()

    # Each packet dataset represent all the parsed packets in a single conversation
    # Each packet is split into a batch based on the byte sequence for training and comes with
    # some metadata and such as categorical and numerical features.
    splits = split_into_conversations(df)
    n_convs = len(splits)
    conv_dfs = [PacketDataset(df, n_convs=n_convs) for df in splits]

    if len(conv_dfs) == 0:
        print(f"Number of conversations must not be zero")
        return

    # Get the categorical and numerical dimensions they will all be the same throughout the conversations
    cat_dims = conv_dfs[0].cat_dims
    numerical_dim = conv_dfs[0].num_dim

    # Define the cross entropy loss model and optimizer
    byte_predictor = NextBytePredictor(cat_dims, numerical_dim, device=DEVICE)

    # Now create the optimizer and criterion
    optimizer = torch.optim.Adam(
        byte_predictor.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss(reduction="mean")

    # Metrics
    best_val_loss = float("inf")
    train_losses = list()
    val_losses = list()

    # Now train over n training epochs
    for epoch in range(N_EPOCHS):

        # Divide each conversation into testing, training, and validation splits
        train, validation, test = split_convs(conv_dfs).values()

        # Enter the training process
        byte_predictor.train()
