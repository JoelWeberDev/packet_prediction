"""
@Author: Joel Weber
@Date: 2025-07-01
@Description: This is the most basic next byte predictor that takes a sequence of text and
learns to predict the next byte in that sequence using an lstm model.

@Notes:
    - Since this is intended to be simple we do not implement batching
    - To both train and generate simply provided a sequence stream

@TODO:

@Packet data integration
The last thing that we are missing is meta data in the model. The task is to still run the next
byte prediction while also updating the meta data whenever a new packet is started. Here are a
couple propositions about how to use meta data:
    1. Create batches based on the packet divisions. Encode the meta data using a MLP.
    2. Repeat the meta data for each byte and encode everything all together.
"""

### Python imports ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from dataclasses import dataclass

### Local imports ###
from preprocessing import load_df, split_into_conversations
from custom_datasets import ByteStream
from CONSTANTS import *
from helper_functions import ConvResults, get_memory, print_update, plot_metrics


class NextByteLSTM(nn.Module):
    # def __init__(self, cat_dims: List[int], num_dims: int, device: str = DEVICE):
    def __init__(self, device: str = DEVICE):
        super().__init__()

        self.byte_embedding = nn.Embedding(VOCAB_DIM, BYTE_EMBED_DIM)

        # Create meta data embeddings
        # def get_embedding_dim(n_cats: int) -> int:
        #     # Google's categorical embedding formuala
        #     return min(MAX_CAT_EMB, round((n_cats * CAT_EMB_SCALAR) ** CAT_EMB_EXPO))

        # # Categorical embeddings
        # self.cat_embeddings = nn.ModuleList(
        #     [
        #         nn.Embedding(cat_size, get_embedding_dim(cat_size))
        #         for cat_size in cat_dims
        #     ]
        # )

        # self.numerical_emb_dims = num_dims
        # self.cat_emb_dims = sum(get_embedding_dim(cat_size) for cat_size in cat_dims)

        # Now create an MLP to handle the metadata

        self.input_size = P_CTX_LEN * BYTE_EMBED_DIM

        self.next_byte_predictor = nn.LSTM(
            input_size=self.input_size,
            hidden_size=P_HIDDEN_SIZE // 2,
            num_layers=P_NUM_LAYERS,
            batch_first=True,
            dropout=P_DROPOUT,
            bidirectional=True,
        )

        # Create the output projections
        self.project_outputs = nn.Linear(P_HIDDEN_SIZE, VOCAB_DIM)

        self.hidden = None

        self.to(device)
        self.device = device

    def forward(self, bytes: torch.Tensor) -> torch.Tensor:
        batch_size, ctx_len = bytes.shape
        assert (
            ctx_len == P_CTX_LEN
        ), f"The byte sequence length {batch_size} != context length {P_CTX_LEN}"

        ctx_embeds = self.byte_embedding(bytes.to(self.device)).reshape(
            batch_size, self.input_size
        )

        output, (hx, cx) = self.next_byte_predictor(ctx_embeds, self.hidden)
        self.hidden = (hx.detach(), cx.detach())

        logits = self.project_outputs(output)

        return logits


if __name__ == "__main__":

    ### Helper functions ###
    def split_convs(
        conv_dfs: List[ByteStream],
    ) -> Dict[str, List[ByteStream]]:
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

    ### Training functions ###
    def run_conv(
        model: NextByteLSTM,
        conv_df: ByteStream,
        optimizer,
        criterion,
        train: bool = True,
    ) -> Tuple[float, float]:
        """
        @Description: This does the batching and training for a given conversation

        @Notes:

        @Returns:
        """
        conv_loss = list()
        conv_acc = list()
        model.hidden = None

        context = list()

        total_cnt = 0
        good_cnt = 0

        window_cnt = 0
        good_window_cnt = 0

        # Go through the packets by batch size and perform the training step for each batch
        while True:

            try:
                byte = next(conv_df)
            except StopIteration:
                break

            if len(context) < P_CTX_LEN:
                context.append(byte)
                continue

            # Create a training batch
            batch = list()
            targets = list()
            if train:

                try:
                    for i in range(P_BATCH_SIZE):
                        batch.append(torch.tensor(context, dtype=torch.long))
                        targets.append(byte)
                        context.append(byte)
                        context = context[-1 * P_CTX_LEN :]
                        byte = next(conv_df)

                except StopIteration:
                    pass
            else:
                targets.append(byte)
                batch.append(torch.tensor(context, dtype=torch.long))

            if len(batch) == 0:
                break

            batch = torch.stack(batch, dim=0).to(DEVICE)
            targets = torch.tensor(targets, dtype=torch.long, device=DEVICE)

            logits = model(batch)
            pred_bytes = logits.argmax(-1)
            loss = criterion(logits, targets)

            if window_cnt != 0 and window_cnt > WINDOW_REFRESH_CNT:
                window_cnt = 0
                good_window_cnt = 0

            total_cnt += batch.shape[0]
            window_cnt += batch.shape[0]

            n_good_pred = int((pred_bytes == targets).sum(dim=-1))
            good_cnt += n_good_pred
            good_window_cnt += n_good_pred

            if train:
                # Back prop the loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            else:
                # Update the context with the prediciton rather than the known
                context += pred_bytes.tolist()  # Should always be a single byte
                context = context[-1 * P_CTX_LEN :]

            global_acc = good_cnt / total_cnt
            window_acc = good_window_cnt / window_cnt
            conv_loss.append(loss.item())
            conv_acc.append(window_acc)

            if DEBUG_MODE:

                # Print some helpful info about the training step
                print_update(
                    batch_num=total_cnt,
                    batch_size=batch.shape[0],
                    loss=loss,
                    window_acc=window_acc,
                    global_acc=global_acc,
                )

        if DEBUG_MODE:
            plot_metrics(
                conv_loss, f"Conversation loss", x_label="Batches", y_label="Batch Loss"
            )
            plot_metrics(
                conv_acc,
                f"Conversation accuracy",
                x_label="Batches",
                y_label="Batch accuracy",
            )

        return (
            np.mean(conv_loss),
            good_cnt / total_cnt if total_cnt > 0 else float("inf"),
        )

    # Get the byte sequence
    def model_train():
        # Get the dataset for the conversation data
        df = load_df()

        # Process the dfs into byte streams
        conv_dfs = [ByteStream(conv_df) for conv_df in split_into_conversations(df)]

        if len(conv_dfs) == 0:
            return

        # cat_dims = conv_dfs.

        # Define the cross entropy loss model and optimizer
        byte_predictor = NextByteLSTM(device=DEVICE)

        # Now create the optimizer and criterion
        optimizer = torch.optim.Adam(
            byte_predictor.parameters(), lr=P_LEARNING_RATE, weight_decay=P_WEIGHT_DECAY
        )

        criterion = nn.CrossEntropyLoss(reduction="mean")

        # Metrics
        best_val_loss = float("inf")
        train_losses = list()
        val_losses = list()
        train_accs = list()
        val_accs = list()

        # Now train over n training epochs
        for epoch in range(N_EPOCHS):
            # Divide each conversation into testing, training, and validation splits
            train, validation, test = split_convs(conv_dfs).values()

            # Set the model in training mode
            byte_predictor.train()
            epoch_loss = 0.0
            epoch_acc = 0.0

            for conv_df in train:
                avg_loss, avg_acc = run_conv(
                    byte_predictor, conv_df, optimizer, criterion, train=True
                )
                epoch_loss += avg_loss
                epoch_acc += avg_acc

            avg_train_loss = epoch_loss / len(conv_dfs)
            train_losses.append(avg_train_loss)

            avg_train_acc = epoch_acc / len(conv_dfs)
            train_accs.append(avg_train_acc)

            # now switch to validation
            print("Switching to validation")
            byte_predictor.eval()
            val_loss = 0.0
            val_acc = 0.0

            with torch.no_grad():
                for conv_df in validation:
                    avg_loss, avg_acc = run_conv(
                        byte_predictor, conv_df, optimizer, criterion, train=False
                    )
                    val_loss += avg_loss
                    val_acc += avg_acc

            avg_val_loss = val_loss / len(conv_dfs)
            val_losses.append(avg_val_loss)

            avg_val_acc = val_acc / len(conv_dfs)
            val_accs.append(avg_val_acc)

            # Model checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": byte_predictor.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                    },
                    f"source_code/checkpoints/model_epoch_{epoch}.pt",
                )

            # Print metrics
            print(f"Epoch {epoch+1}/{N_EPOCHS}:")
            print(f"  Training Loss: {avg_train_loss:.4f}")
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            print(f"  Training Loss: {avg_train_acc:.4f}")
            print(f"  Validation Loss: {avg_val_acc:.4f}")

            # Early stopping check
            if len(val_losses) > PATIENCE:
                if all(val_losses[-PATIENCE:] > best_val_loss):
                    print("Early stopping triggered")
                    break

            # Process the dfs into byte streams
            conv_dfs = [ByteStream(conv_df) for conv_df in split_into_conversations(df)]

    model_train()
