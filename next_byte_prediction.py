"""
@Author: Joel Weber
@Date: 2025-07-01
@Description: This is the most basic next byte predictor that takes a sequence of text and
learns to predict the next byte in that sequence using an lstm model.

@Notes:
    - Since this is intended to be simple we do not implement batching
    - To both train and generate simply provided a sequence stream

@TODO:
    - Implement auto regressive training rather than trainer forcing. This way we learn how to
    properly predict packets.

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
from custom_datasets import ByteStream, PacketDataset
from CONSTANTS import *
from helper_functions import ConvResults, get_memory, print_update, plot_metrics


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


class NextByteLSTM(nn.Module):
    def __init__(self, cat_dims: List[int], num_dims: int, device: str = DEVICE):
        super().__init__()

        self.byte_embedding = nn.Embedding(VOCAB_DIM, BYTE_EMBED_DIM)

        # Create meta data embeddings
        def get_embedding_dim(n_cats: int) -> int:
            # Google's categorical embedding formuala
            return min(MAX_CAT_EMB, round((n_cats * CAT_EMB_SCALAR) ** CAT_EMB_EXPO))

        # Categorical embeddings
        self.cat_embedder = nn.ModuleList(
            [
                nn.Embedding(cat_size, get_embedding_dim(cat_size))
                for cat_size in cat_dims
            ]
        )

        self.numerical_emb_dim = num_dims
        self.cat_emb_dim = sum(get_embedding_dim(cat_size) for cat_size in cat_dims)

        self.metadata_dim = self.cat_emb_dim + self.numerical_emb_dim

        # Now create an MLP to handle the metadata
        self.metadata_mpl = nn.Sequential(
            nn.Linear(self.metadata_dim, P_METADATA_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(P_METADATA_DROPOUT),
            nn.Linear(P_METADATA_HIDDEN_DIM, P_METADATA_OUTPUT_DIM),
        )

        self.ctx_dim = P_CTX_LEN * BYTE_EMBED_DIM

        self.input_size = self.ctx_dim + P_METADATA_OUTPUT_DIM

        self.next_byte_predictor = nn.LSTM(
            input_size=self.input_size,
            hidden_size=P_HIDDEN_SIZE // 2,
            num_layers=P_NUM_LAYERS,
            batch_first=True,
            dropout=P_DROPOUT,
            bidirectional=True,
        )

        self.input_norm = nn.LayerNorm(self.input_size)

        # Create the output projections
        self.project_outputs = nn.Linear(P_HIDDEN_SIZE, VOCAB_DIM)

        self.hidden = None

        self.to(device)
        self.device = device

        # Embed zeros to ensure that the metadata_emb property is initialized
        self.embed_meta_data(
            torch.zeros(self.cat_emb_dim, dtype=torch.long, device=device),
            torch.zeros(self.numerical_emb_dim, dtype=torch.long, device=device),
        )

    def embed_meta_data(self, cat_features: torch.Tensor, num_features: torch.Tensor):
        """
        @Description: Updates meta data embeddings in the separate mlp

        @Notes:
            - The meta data cannot be batched, it must be a singular set of meta data

        """
        cat_features = cat_features.to(self.device)
        num_features = num_features.to(self.device)

        # Embed the categorical features using the cat embeddings from above
        self.cat_emb = torch.cat(
            [
                cat_embedder(cat_f)
                for cat_f, cat_embedder in zip(cat_features, self.cat_embedder)
            ],
            dim=-1,
        )

        self.num_emb = num_features

        self.metadata_emb = (
            self.metadata_mpl(torch.cat([self.cat_emb, num_features], dim=-1))
        ) * P_METADATA_SCALE

    def forward(
        self,
        bytes: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, ctx_len = bytes.shape
        assert (
            ctx_len == P_CTX_LEN
        ), f"The byte sequence length {batch_size} != context length {P_CTX_LEN}"

        ctx_embeds = self.byte_embedding(bytes.to(self.device)).reshape(
            batch_size, self.ctx_dim
        )

        meta_rep = self.metadata_emb.unsqueeze(0).expand(batch_size, -1)
        input_emb = self.input_norm(torch.cat([ctx_embeds, meta_rep], dim=-1))

        output, (hx, cx) = self.next_byte_predictor(input_emb, self.hidden)
        self.hidden = (hx.detach(), cx.detach())

        logits = self.project_outputs(output)

        return logits


if __name__ == "__main__":

    ### Helper functions ###
    def split_convs(
        conv_dfs: List[PacketDataset],
    ) -> Dict[str, List[PacketDataset]]:
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
        conv_df: PacketDataset,
        optimizer,
        criterion,
        train: bool = True,
    ) -> Tuple[float, float]:
        """
        @Description: This takes the next packet from the data set and uses that to train or
        generate the next packet.

        @Notes:
            - The context is simply stored as a fixed length of bytes
            - The batch size is determined by the sequence length
            - New meta data is encoded for each packet

        @Returns:
        """
        conv_loss = list()
        conv_acc = list()
        model.hidden = None

        context = list()

        tot_cnt = 0
        tot_good_cnt = 0

        window_cnt = 0
        good_window_cnt = 0

        # Go through the packets by batch size and perform the training step for each batch
        while True:
            try:
                cur_packet = next(conv_df)
            except StopIteration:
                break

            # Encode the meta data
            cat_f = cur_packet.cat_features
            num_f = cur_packet.numerical_features

            # Create a training batch
            batch = list()
            targets = list()
            batch_num = 1
            batch_loss = 0
            n_good = 0

            context.append(SOS)
            if len(context) > P_CTX_LEN:
                context = context[-1 * P_CTX_LEN :]

            if len(cur_packet.payload) == 0:
                continue

            # Embed the metadata features into the model
            # Embed zeros to ensure that the metadata_emb property is initialized
            # model.embed_meta_data(
            #     torch.zeros(model.cat_emb_dim, dtype=torch.long, device=DEVICE),
            #     torch.zeros(model.numerical_emb_dim, dtype=torch.long, device=DEVICE),
            # )
            model.embed_meta_data(cat_f, num_f)

            # Ensure we have ample context
            for byte in cur_packet.payload:
                if len(context) < P_CTX_LEN:
                    context.append(byte)
                    continue

                if train:
                    batch.append(torch.tensor(context, dtype=torch.long))
                    targets.append(byte)
                    context.append(byte)
                else:
                    # Get the predicted byte
                    logits = model(
                        torch.tensor([context], dtype=torch.long, device=DEVICE)
                    )
                    loss = criterion(
                        logits, torch.tensor([byte], dtype=torch.long, device=DEVICE)
                    )
                    batch_loss += loss.item()

                    pred_byte = int(logits.argmax(-1)[0])
                    context.append(pred_byte)

                    if byte == pred_byte:
                        n_good += 1

                context = context[-1 * P_CTX_LEN :]

            if train:
                batch = torch.stack(batch, dim=0).to(DEVICE)
                targets = torch.tensor(targets, dtype=torch.long, device=DEVICE)

                logits = model(batch)
                pred_bytes = logits.argmax(-1)
                loss = criterion(logits, targets)

                n_good = int((pred_bytes == targets).sum(dim=-1))

                print(f"Pred bytes: {pred_bytes}\ntarget bytes: {targets}\n")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_loss = loss.item()

            tot_cnt += len(cur_packet.payload)
            tot_good_cnt += n_good
            conv_loss.append(batch_loss)
            conv_acc.append(n_good / len(cur_packet.payload))
            batch_num += 1

            if DEBUG_MODE:

                # Print some helpful info about the training step
                print_update(
                    batch_num=batch_num,
                    batch_size=len(cur_packet.payload),
                    loss=batch_loss,
                    batch_acc=conv_acc[-1],
                    global_acc=tot_good_cnt / tot_cnt,
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
            float(np.mean(conv_loss)),
            tot_good_cnt / tot_cnt if tot_cnt > 0 else float("inf"),
        )

    # Get the byte sequence
    def model_train():
        # Get the dataset for the conversation data
        df = load_df()

        # Process the dfs into byte streams
        splits = split_into_conversations(df)
        conv_dfs = [PacketDataset(conv_df, n_convs=len(splits)) for conv_df in splits]

        if len(conv_dfs) == 0:
            return

        cat_dims = conv_dfs[0].cat_dims
        num_dims = conv_dfs[0].num_dims

        # Define the cross entropy loss model and optimizer
        byte_predictor = NextByteLSTM(
            cat_dims=cat_dims, num_dims=num_dims, device=DEVICE
        )

        # Now create the optimizer and criterion
        optimizer = torch.optim.Adam(
            byte_predictor.parameters(), lr=P_LEARNING_RATE, weight_decay=P_WEIGHT_DECAY
        )

        # criterion = nn.CrossEntropyLoss(reduction="mean")
        criterion = LabelSmoothingCrossEntropy(smoothing=P_SMOOTHING)

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

            print(f"Epoch {epoch+1}/{N_EPOCHS}:")
            print(f"  Training Loss: {avg_train_loss:.4f}")
            print(f"  Training Acc: {avg_train_acc:.4f}")

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
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            print(f"  Validation Acc: {avg_val_acc:.4f}")

            # Early stopping check
            if len(val_losses) > PATIENCE:
                if all(val_losses[-PATIENCE:] > best_val_loss):
                    print("Early stopping triggered")
                    break

            # Process the dfs into byte streams
            conv_dfs = [
                PacketDataset(conv_df, n_convs=len(splits)) for conv_df in splits
            ]

    model_train()
