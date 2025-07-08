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
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from dataclasses import dataclass
from copy import deepcopy

### Local imports ###
from preprocessing import load_df, split_into_conversations, load_dfs_from_dir
from custom_datasets import ByteStream, PacketDataset
from CONSTANTS import *
from helper_functions import (
    ConvResults,
    EpochResults,
    get_memory,
    print_update,
    plot_metrics,
)


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

    ### Cross function variables ###
    conv_list = list()
    split_dict = {
        "train": [],
        "val": [],
        "test": [],
    }

    ### Helper functions ###
    def update_split_dict():
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

    ### Training functions ###
    def run_payload(
        model: NextByteLSTM,
        context: List[int],
        payload: List[int],
        force_teacher: bool = True,
        random_mask_ctx: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, int]:

        # Add the start of packet indication
        context.append(SOS)
        if len(context) > P_CTX_LEN:
            context.pop(0)
            # context = context[-1 * P_CTX_LEN :]

        assert (
            len(context) <= P_CTX_LEN
        ), f"The context length of {len(context)} must not exceed {P_CTX_LEN}"

        # Create a training batch
        n_good = 0

        # Make allowence for accumulating context
        batch_size = max(len(payload) - P_CTX_LEN + len(context), 0)
        batch_logits = torch.zeros(
            (batch_size, VOCAB_DIM), dtype=torch.float32, device=DEVICE
        )
        batch_preds = torch.zeros((batch_size), dtype=torch.long, device=DEVICE)
        batch_targets = torch.zeros((batch_size), dtype=torch.long, device=DEVICE)

        i = 0
        # Ensure we have ample context
        for byte in payload:
            if len(context) < P_CTX_LEN:
                context.append(byte)
                continue

            # Get the predicted byte
            ctx_tensor = torch.tensor([context], dtype=torch.long)

            if random_mask_ctx:
                mask_inds = np.random.choice(
                    range(len(context)), round(P_CTX_MASK_PERC * len(context))
                )
                ctx_tensor[0, mask_inds] = MASK

            ctx_tensor.to(model.device)

            logits = model(ctx_tensor)

            pred_byte = int(logits.argmax(-1)[0])
            if force_teacher:
                context.append(byte)
            else:
                context.append(pred_byte)

            batch_logits[i, :] = logits
            batch_targets[i] = byte
            batch_preds[i] = pred_byte

            if byte == pred_byte:
                n_good += 1

            # Ensure our context length remains the same
            context.pop(0)

            i += 1

        return batch_size, batch_logits, batch_targets, batch_preds, n_good

    def run_conv(
        model: NextByteLSTM,
        conv_df: PacketDataset,
        optimizer,
        criterion,
        train: bool = True,
        show_plots: bool = DEBUG_MODE,
    ) -> Tuple[float, float]:
        """
        @Description: This takes the next packet from the data set and uses that to train or
        generate the next packet.

        @Notes:
            - The context is simply stored as a fixed length of bytes
            - The batch size is determined by the sequence length
            - New meta data is encoded for each packet
            - We do byte by byte training rather than teacher forcing. Since the model is auto
            regressive, any single error will throw the entire prediction off. Therefore the
            model must be trained in the same manner that it will infer.
        @Returns:
        """
        conv_loss = list()
        conv_acc = list()

        # Keep 2 copies of the context and hidden states for the purpose of updating the
        # hidden state to the true content once the packet has been predicted
        pred_context = list()
        true_context = list()

        true_hidden = None

        tot_cnt = 0
        tot_good_cnt = 0
        batch_num = 1

        # Go through the packets by batch size and perform the training step for each batch
        while True:
            try:
                cur_packet = next(conv_df)
            except StopIteration:
                break

            # Encode the meta data
            cat_f = cur_packet.cat_features
            num_f = cur_packet.numerical_features

            # Embed the metadata features into the model
            model.embed_meta_data(cat_f, num_f)

            if len(cur_packet.payload) == 0:
                continue

            # Run th model to get the predictions and loss with no teacher forcing
            model.hidden = deepcopy(true_hidden)
            batch_size, batch_logits, batch_targets, batch_preds, n_good = run_payload(
                model,
                pred_context,
                cur_packet.payload,
                force_teacher=False,
                random_mask_ctx=True,
            )

            # Run the model with trainer forcing to update the true context
            model.hidden = true_hidden
            _, true_logits, true_targets, true_preds, n_true_good = run_payload(
                model,
                true_context,
                cur_packet.payload,
                force_teacher=True,
                random_mask_ctx=False,
            )

            if batch_size == 0:
                continue

            # Use the loss from the auto regressive generation since it best matches the validation process
            loss = criterion(batch_logits, batch_targets)
            batch_loss = loss.item()

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            tot_cnt += batch_size
            tot_good_cnt += n_good
            conv_loss.append(batch_loss)
            conv_acc.append(n_good / batch_size)
            batch_num += 1

            # Now that we have completed a batch update the context with the
            # legitimate bytes
            # upd_len = min(len(cur_packet.payload), P_CTX_LEN)
            # context[-1 * upd_len :] = cur_packet.payload[-1 * upd_len :]

            if DEBUG_MODE:
                print(f"Pred bytes: {batch_preds}\ntarget bytes: {batch_targets}\n")
                # Print some helpful info about the training step
                print_update(
                    batch_num=batch_num,
                    batch_size=len(cur_packet.payload),
                    loss=batch_loss,
                    batch_acc=conv_acc[-1],
                    global_acc=tot_good_cnt / tot_cnt,
                )

            if (train and batch_num > P_TRAIN_INTERRUPT) or (
                not train and batch_num > P_VAL_INTERRUPT
            ):
                break

        if show_plots:
            mode = "Train" if train else "Validation"
            plot_metrics(
                conv_loss,
                f"Conv {conv_df.conv_num} loss in ({mode})",
                x_label="Batches",
                y_label="Batch Loss",
            )
            plot_metrics(
                conv_acc,
                f"Conv {conv_df.conv_num} accuracy in ({mode})",
                x_label="Batches",
                y_label="Batch accuracy",
            )

        return (
            float(np.mean(conv_loss)),
            tot_good_cnt / tot_cnt if tot_cnt > 0 else float("inf"),
        )

    def train_epoch(
        conv_dfs: List[PacketDataset],
        model: NextByteLSTM,
        optimizer,
        show_plots: bool = DEBUG_MODE,
    ) -> EpochResults:
        results = EpochResults()
        criterion = LabelSmoothingCrossEntropy(smoothing=P_SMOOTHING)

        # Divide each conversation into testing, training, and validation splits
        train, validation, test = split_convs(conv_dfs).values()

        print(f"split dict: {split_dict}")

        # Set the model in training mode
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        for conv_df in train:
            avg_loss, avg_acc = run_conv(
                model, conv_df, optimizer, criterion, train=True, show_plots=show_plots
            )
            epoch_loss += avg_loss
            epoch_acc += avg_acc

        results.avg_train_loss = epoch_loss / len(conv_dfs)

        results.avg_train_acc = epoch_acc / len(conv_dfs)

        # now switch to validation
        print("Switching to validation")
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for conv_df in validation:
                avg_loss, avg_acc = run_conv(
                    model,
                    conv_df,
                    optimizer,
                    criterion,
                    train=False,
                    show_plots=show_plots,
                )
                val_loss += avg_loss
                val_acc += avg_acc

        results.avg_val_loss = val_loss / len(conv_dfs)

        results.avg_val_acc = val_acc / len(conv_dfs)

        return results

    # Get the byte sequence
    def model_train(csv_dir: str):

        # Metrics
        best_val_loss = float("inf")
        train_losses = list()
        val_losses = list()
        train_accs = list()
        val_accs = list()

        # Declare empty model, optimizer and criterion
        byte_predictor = None
        optimizer = None

        # Now train over n training epochs
        for epoch in range(N_EPOCHS):
            dfs = load_dfs_from_dir(csv_dir=csv_dir)
            for df in dfs:
                # Get the conversations splits
                splits = split_into_conversations(df, conv_list=conv_list)

                conv_dfs = [
                    PacketDataset(conv_df, n_convs=len(conv_list)) for conv_df in splits
                ]

                # Since we now have the features and dimensions we can initialize the model
                if byte_predictor is None:
                    cat_dims = conv_dfs[0].cat_dims
                    num_dims = conv_dfs[0].num_dims
                    byte_predictor = NextByteLSTM(
                        cat_dims=cat_dims, num_dims=num_dims, device=DEVICE
                    )
                    optimizer = torch.optim.Adam(
                        byte_predictor.parameters(),
                        lr=P_LEARNING_RATE,
                        weight_decay=P_WEIGHT_DECAY,
                    )

                results = train_epoch(
                    conv_dfs, byte_predictor, optimizer=optimizer, show_plots=False
                )

                print(f"Epoch {epoch+1}/{N_EPOCHS}:")
                print(f"  Training Loss: {results.avg_train_loss:.4f}")
                print(f"  Training Acc: {results.avg_train_acc:.4f}")

                train_losses.append(results.avg_train_loss)
                train_accs.append(results.avg_train_acc)
                val_losses.append(results.avg_val_loss)
                val_accs.append(results.avg_val_acc)

                # Model checkpointing
                if results.avg_val_loss < best_val_loss:
                    best_val_loss = results.avg_val_loss
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": byte_predictor.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": results.avg_train_loss,
                            "val_loss": results.avg_val_loss,
                        },
                        f"source_code/checkpoints/model_epoch_{epoch}.pt",
                    )

                # Print metrics
                print(f"Epoch {epoch+1}/{N_EPOCHS}:")
                print(f"  Validation Loss: {results.avg_val_loss:.4f}")
                print(f"  Validation Acc: {results.avg_val_acc:.4f}")

                # Early stopping check
                if len(val_losses) > PATIENCE:
                    if all([v > best_val_loss for v in val_losses[-PATIENCE:]]):
                        print("Early stopping triggered")
                        break

                # Process the dfs into packet datasets
                conv_dfs = [
                    PacketDataset(conv_df, n_convs=len(splits)) for conv_df in splits
                ]

        # Plot the metrics over the training process
        plot_metrics(
            train_losses,
            title=f"Overall training loss",
            x_label="epoch",
            y_label="Loss",
        )
        plot_metrics(
            train_accs,
            title=f"Overall training accuracy",
            x_label="epoch",
            y_label="Accuracy",
        )
        plot_metrics(
            val_losses,
            title=f"Overall validation loss",
            x_label="epoch",
            y_label="Loss",
        )
        plot_metrics(
            val_accs,
            title=f"Overall validation accuracy",
            x_label="epoch",
            y_label="Accuracy",
        )

    ### Training entry point ###
    # csv_dir = "datasets/mqtt-data/kaggle_mqtt_set/Data/PCAP/legit_cap_split/legtimate_w1-1_split"
    csv_dir = (
        "datasets/mqtt-data/kaggle_mqtt_set/Data/PCAP/legit_cap_split/small_sample"
    )
    model_train(csv_dir=csv_dir)
