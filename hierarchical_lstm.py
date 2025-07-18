"""
@Author: Joel Weber
@Date: 2025-06-20
@Description:

@Notes:
    Workflow:
        Conversational LSTM
            - Ingests the conversation packet by packet.
            - Stores a conversation hidden state
            - Will call the next byte lstm to generate the next packet
            - Only accepts one packet at a time

@Questions:
    - What does the packing actually do?
    - Are hidden states persisent throughout calls? Do I need to store that hidden state and
    pass it in each time I use the LSTM?
    - How much should I rely on memory of internal states and how much on the context?
    - How to create an enum in python
    - If I am running the model on an embedded system, what is a reasonable parameter count?
    - Is it worth zeroing out the embeddings for null characters?
    - Is projecting to eliminate special characters a valid way to include the without letting
    the model actually predict them?


@TODO:
    - Refactor the very broken next packet predicter
    - Fix the byte sequence encoding in the packet encoder

"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Local imports
from CONSTANTS import *
from preprocessing import load_df, load_dfs_from_dir, split_into_conversations
from custom_datasets import PacketDataset, ParsedPacket
from helper_functions import (
    split_dict,
    conv_list,
    split_convs,
    google_get_embedding_dim,
    sample_with_temperature,
    plot_metrics,
    ConvResults,
    EpochResults,
    LabelSmoothingCrossEntropy,
)

### Globals ###
g_cur_temperature = P_INITIAL_TEMPERATURE


class PacketEncoder(nn.Module):
    """Encodes individual packet (metadata + payload) into fixed representation"""

    def __init__(self, cat_dims: List[int], num_dims: int, device: str = DEVICE):
        super().__init__()

        # Byte embedding
        self.byte_embedder = nn.Embedding(VOCAB_DIM, BYTE_EMBED_DIM)

        # Categorical embeddings
        self.cat_embedder = nn.ModuleList(
            [
                nn.Embedding(cat_size, google_get_embedding_dim(cat_size))
                for cat_size in cat_dims
            ]
        )

        # Calculate input dimension for packet LSTM
        cat_embed_dim = sum(google_get_embedding_dim(cat_size) for cat_size in cat_dims)

        # packet_lstm_input_dim = byte_embed_dim + cat_embed_dim + numerical_dim
        packet_lstm_input_dim = BYTE_EMBED_DIM

        # Packet-level LSTM (processes bytes within packet)
        self.packet_lstm = nn.LSTM(
            input_size=packet_lstm_input_dim,
            hidden_size=H_PACKET_REP_DIM // 2,  # Will be bidirectional
            num_layers=H_PACKET_ENC_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=H_PACKET_ENC_DROPOUT,
        )

        # Additional MLP to process metadata separately
        self.metadata_mlp = nn.Sequential(
            nn.Linear(cat_embed_dim + num_dims, H_PACKET_REP_DIM // 4),
            nn.ReLU(),
            nn.Dropout(METADATA_MLP_DROPOUT),
        )

        # Combine packet content + metadata
        self.packet_combiner = nn.Sequential(
            nn.Linear(H_PACKET_REP_DIM + H_PACKET_REP_DIM // 4, H_PACKET_REP_DIM),
            nn.ReLU(),
            nn.Dropout(H_PACKET_COMBINER_DROPOUT),
        )

    def forward(
        self,
        payload: torch.Tensor,
        cat_features: torch.Tensor,
        num_features: torch.Tensor,
    ):
        """
        @Description: Forward pass for the single packet embedding and prediciton LSTM model.
        This takes a batch of inputs in the byte sequence, and produces the output predictions
        for that input. Here we also are given the corresponding attention masks for the
        each byte sequence. The attention mask must have the same shape as the byte sequence.

        @Notes:
            - This expects a payload with a context length of H_PAYLOAD_LEN


        @Returns:
        """
        assert (
            len(payload.shape) == 1
        ), f"The payload must be 1D, not {len(payload.shape)}"

        # Embed the categorical features using the cat embeddings from above
        cat_emb = torch.cat(
            [
                cat_embedder(cat_f)
                for cat_f, cat_embedder in zip(cat_features, self.cat_embedder)
            ],
            dim=-1,
        )

        # Embed bytes
        byte_embeds = self.byte_embedder(payload)  # [payload_len, embed_dim]

        # Now pack the embeddings using the attention mask

        # Process through packet LSTM
        lstm_output, _ = self.packet_lstm(byte_embeds)

        # Take final hidden state as packet representation
        packet_content_repr = lstm_output[-1, :]  # [packet_repr_dim]

        assert packet_content_repr.shape == (
            H_PACKET_REP_DIM,
        ), f"The packet shape of {packet_content_repr.shape} must equal {(H_PACKET_REP_DIM,)}"

        # Process metadata separately
        metadata_input = torch.cat([cat_emb, num_features], dim=-1)
        metadata_repr = self.metadata_mlp(metadata_input)

        # Combine representations
        packet_repr = self.packet_combiner(
            torch.cat(
                [packet_content_repr, metadata_repr],
                dim=-1,
            )
        )

        return packet_repr


class ConversationLSTM(nn.Module):
    """Maintains conversation state across packets"""

    def __init__(self):
        super().__init__()

        self.conversation_lstm = nn.LSTM(
            input_size=H_PACKET_REP_DIM,
            hidden_size=H_CONVERSATIONAL_HIDDEN_DIM,
            num_layers=H_CONVERSATIONAL_LAYERS,
            batch_first=True,
            dropout=H_CONVERSATIONAL_LSTM_DROPOUT,
        )

        self.hidden = None

    def forward(self, packet_representations):
        """
        @Args:
            packet_representations: [batch, num_packets, packet_repr_dim]
            hidden_state: Previous conversation state (h, c)
        @Returns:
            conversation_outputs: [batch, num_packets, conversation_hidden_dim]
            final_hidden_state: Updated conversation state
        """
        conversation_outputs, (h, c) = self.conversation_lstm(
            packet_representations, self.hidden
        )

        self.hidden = (h.detach(), c.detach())
        return conversation_outputs


class NextPacketPredictor(nn.Module):
    """
    @Description: We take a context history of past packets, the target packet's meta data
    and the past bytes if there are any to autoregressively predict the next byte in the
    sequence.
        Input for next byte: CAT_EMBED_DIMS + NUM_EMBED_DIMS + CONVERSATION_CONTEXT_DIMS + (SEQ_LEN * BYTE_EMBED_DIMS)

    Since the byte context size is fixed, but we are allowing for variable length input
    sequences we need to use the NULL token for padding to the FRONT!!. In the sequence the
    byte at the very end is considered the most recent one.

    @Notes:
        - All the input except for the sequence embeddings remains the same.
        - We will not enforce a fixed sequence length, but rather pack padded sequences
        - In addition to the 256 byte vocabulary we also enlist a number of special tokens
        each defined in the CONSTANTS.py file.
        - The SOS characeter at the start of each packet counts in the sequence length count
        - Target payloads do not need to be passed with lenths of MAX_SEQ_LEN, rather just
        padded with null character to meet the length of the longest payload
        - When processing the the predictions we project down to 256 for the bytes only and not
        the special tokens. ***

    @Returns:
    """

    def __init__(self, cat_dims: List[int], numerical_dim=0):
        super().__init__()

        # Categorical embeddings
        self.cat_embedder = nn.ModuleList(
            [
                nn.Embedding(cat_size, google_get_embedding_dim(cat_size))
                for cat_size in cat_dims
            ]
        )

        # Calculate input dimension for packet LSTM
        cat_embed_dim = sum(google_get_embedding_dim(cat_size) for cat_size in cat_dims)

        # Combine conversation context with next packet metadata
        self.input_size = (
            H_CONVERSATIONAL_HIDDEN_DIM
            + cat_embed_dim
            + numerical_dim
            + (H_BYTE_CONTEXT_LEN * BYTE_EMBED_DIM)
        )

        # Decoder LSTM for payload generation
        self.decoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=H_CONVERSATIONAL_HIDDEN_DIM,
            num_layers=H_NEXT_PACKET_LAYERS,
            batch_first=True,
            dropout=H_NEXT_PACKET_DROPOUT,
        )

        # Output projection on just the 256 bytes excluding the special tokens
        self.output_projection = nn.Linear(H_CONVERSATIONAL_HIDDEN_DIM, VOCAB_DIM)

        # Byte embedding for decoder
        self.byte_embedding = nn.Embedding(VOCAB_DIM, BYTE_EMBED_DIM)

        self.hidden = None

    def embed_cats(self, cat_fs: torch.Tensor):
        # Embed next packet categorical features
        cat_embeds = torch.cat(
            [
                embedding_layer(cat_f)
                for cat_f, embedding_layer in zip(cat_fs, self.cat_embedder)
            ],
            dim=-1,
        )
        return cat_embeds

    def forward(
        self,
        embedded_conversation_context: torch.Tensor,  # embedded context
        next_packet_categorical: torch.Tensor,
        next_packet_numerical: torch.Tensor,
        target_payload: None | torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        @Args:
            conversation_context: [batch, conversation_hidden_dim]
            next_packet_categorical: [batch, num_categorical]
            next_packet_numerical: [batch, numerical_dim]
            target_payload: [batch, payload_length] - for teacher forcing during training
        @Notes:
            - The mqtt.len is the third arguement in the numerical features
            - This can only handle a single packet a time. Not multiple batches
        """
        # TEST: repeating process for cat embedding
        cat_embeds = self.embed_cats(next_packet_categorical)

        # Combine context with next packet metadata as predictors
        context = torch.cat(
            [embedded_conversation_context, cat_embeds, next_packet_numerical], dim=-1
        )

        if target_payload is None:
            # Inference mode - autoregressive generation
            logits, preds = self._generate_payload(context)
        else:
            # Training mode - teacher forcing
            logits = self._train_forward(context, target_payload)
            preds = logits.argmax(dim=-1)

        return logits, preds

    def _train_forward(
        self, context: torch.Tensor, target_payload: torch.Tensor
    ) -> torch.Tensor:
        """
        @Description: Uses the batch contexts along the prior embedded next packet features
        to predict the entire sequence of next bytes for the target packet.

        @Notes:
            context: [batch_size, context_length, packet_embed_size]
            target_payload: [batch_size, payload_length]

        @TODO:
            - Should I use the mqtt.length to explicitly cut the prediciton off or rather
            allow the model to learn that it should stop at mqtt.length

        @Returns:
            logits: torch.tensor([]) -> [payload_length, BYTE_VOCAB_DIMS]
        """
        assert len(target_payload.shape) == 1, f"The target payload must be 1D"
        assert len(context.shape) == 1, f"The context tensor must be 1D"

        payload_len = target_payload.shape[0]
        lstm_input = torch.empty(
            (payload_len, self.input_size), dtype=torch.float32, device=DEVICE
        )

        padded_payload_emb = self.byte_embedding(
            torch.cat(
                [
                    torch.tensor(
                        [MASK] * (H_BYTE_CONTEXT_LEN - 1) + [SOS],
                        dtype=torch.long,
                        device=DEVICE,
                    ),
                    target_payload,
                ]
            )
        )

        for i in range(payload_len):
            # Create the embeddings for the context
            lstm_input[i, :] = torch.cat(
                [
                    context,
                    padded_payload_emb[i : i + H_BYTE_CONTEXT_LEN].flatten(),
                ],
                dim=-1,
            )  # [ctx_len + max_len * BYTE_EMBED_DIMS]

        # Process through decoder LSTM
        output, (h, c) = self.decoder_lstm(lstm_input, self.hidden)
        self.hidden = (h.detach(), c.detach())

        # Project to vocabulary
        logits = self.output_projection(output)

        return logits  # [payload_size, BYTE_VOCAB_SIZE]

    def _generate_payload(
        self, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        @Description: Goes through autoregressively predicting the packet payload byte by byte

        @Notes:
            - The length of the payload must be provided. Typically this is contained within
            the meta data.
            - context: torch.tensor [H_CONVERSATION_CTX_LEN]

        @TEST:
            - Does this padded sequnce only predict for a single byte? What is the output shape?

        @Returns: List[torch.Tensor]
        """
        # Autoregressively predict the next byte in the packet, adding it to the context each time
        assert len(context.shape) == 1, f"The context tensor must be 1D"
        context_len = context.shape[0]

        padded_pred_payload = torch.tensor(
            [MASK] * (H_BYTE_CONTEXT_LEN - 1) + [SOS] + [MASK] * context_len,
            dtype=torch.long,
        )

        all_logits = torch.empty(
            (context_len, VOCAB_DIM), dtype=torch.float32, device=DEVICE
        )

        for i in range(context_len):
            # Encode only the prior bytes
            byte_embeds = (
                self.byte_embedding(padded_pred_payload[i : i + H_BYTE_CONTEXT_LEN])
                .to(context.device)
                .flatten()
            )  # [H_BYTE_CONTEXT_LEN, BYTE_EMBED_SIZE]

            # make a copy of the context for the current timestamp
            context_step = context  # add another dimension to the tensor

            lstm_input = torch.cat(
                [context_step, byte_embeds], dim=1
            )  # [full_context_size]

            # Get the LSTM output for the given input
            output, (h, c) = self.decoder_lstm(lstm_input, self.hidden)
            self.hidden = (h.detach(), c.detach())

            logits = self.output_projection(
                output
            )  # Only gets the very last prediction

            # Select the most likely byte for each
            # pred_byte = logits.argmax(dim=-1)
            pred_byte = sample_with_temperature(logits, temp=H_TEMP)

            padded_pred_payload[i + H_BYTE_CONTEXT_LEN] = pred_byte

            all_logits[i, :] = logits

        # Now construct the sequence of output payload predictions
        return all_logits, padded_pred_payload[H_BYTE_CONTEXT_LEN:]


class HeirarchicalMQTTModel(nn.Module):
    """Complete hierarchical model for MQTT conversation modeling"""

    def __init__(
        self, categorical_dims: List[int], numerical_dim: int, device: str = DEVICE
    ):
        super().__init__()

        self.packet_encoder = PacketEncoder(
            cat_dims=categorical_dims, num_dims=numerical_dim
        )

        self.conversation_lstm = ConversationLSTM()

        self.next_packet_predictor = NextPacketPredictor(
            cat_dims=categorical_dims, numerical_dim=numerical_dim
        )

        self.to(device)

    def forward(
        self,
        context_packets_emb: torch.Tensor,
        next_cat: torch.Tensor,
        next_numerical: torch.Tensor,
        target_payload: torch.Tensor | None = None,
    ):
        """
        @Description: Computes the forward step for for a batch of contexts and packets
        This can be used either for training or next packet generation.

        @Notes:
            - Any parsed packet must have the full sequence length provided.

        @Args:
            conversation packets: torch.tensor(ParsedPackets) [batch_size, context length]
            next_cat: torch.tensor(list(int)) [batch_size, 1]
            next_numerical: torch.tensor(list(int)) [batch_size, 1]
            target_payload: torch.tensor(list(list(int))) [batch_size, variable]

        @Returns: logits distribution of next packet prediction.
        """
        assert context_packets_emb.shape == (
            H_PACKET_CTX,
            H_PACKET_REP_DIM,
        ), f"The conversation packets must have shape {(H_PACKET_CTX, H_PACKET_REP_DIM)} not {context_packets_emb.shape}"

        # Embedded packet shape: [byte_embeddings + cat_embed_dim + num_embed_dim]
        # individual embedding shape: [context_length, embedded_packet_length]

        # Process through conversation LSTM
        conversation_outputs = self.conversation_lstm(context_packets_emb)

        # Use final conversation state to predict next packet
        final_conversation_context = conversation_outputs[-1, :]

        # Predict next packet payload
        logits, preds = self.next_packet_predictor.forward(
            final_conversation_context,
            next_cat,
            next_numerical,
            target_payload,
        )

        return logits, preds

    def reset_hidden(self):
        self.conversation_lstm.hidden = None
        self.next_packet_predictor.hidden = None


### Training section ###
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


def run_conv(
    model: HeirarchicalMQTTModel,
    conv_df: PacketDataset,
    optimizer,
    criterion,
    train: bool = True,
    show_plots: bool = DEBUG_MODE,
) -> Tuple[float, float]:
    """
    @Description: Takes an conversation and runs the training or validation on that packet

    @Notes:
        - Each packet requires an embeded context of other packets
        - The packet itself is predicted byte by byte either auto regressively or by trainer forcing
        - At the moment we only work with batch sizes of one packet

    @Returns:
    """
    conv_loss = list()
    conv_acc = list()

    context = list()

    model.reset_hidden()

    byte_cnt = 0
    correct_byte_cnt = 0
    batch_num = 1
    mode = "Train" if train else "Validation"
    end_reached = False

    # Go through the packets by batch size and perform the training step for each batch
    while not end_reached:
        try:
            cur_packet = next(conv_df)
        except StopIteration:
            end_reached = True
            break

        # Predict the next packet
        target_payload = torch.tensor([SOS] + cur_packet.payload, dtype=torch.long).to(
            device=DEVICE
        )

        if len(context) == H_PACKET_CTX:
            if len(cur_packet.payload) == 0:
                continue

            logits, preds = model.forward(
                torch.stack(context).to(device=DEVICE),
                cur_packet.cat_features.to(device=DEVICE),
                cur_packet.numerical_features.to(device=DEVICE),
                target_payload=target_payload if train else None,
            )

            loss = criterion(logits, target_payload)
            packet_loss = loss.item()

            if train:
                # Compute the loss
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            else:
                pass

            packet_good_cnt = 0
            for pred, correct in zip(preds, cur_packet.payload):
                if pred == correct:
                    packet_good_cnt += 1
                    correct_byte_cnt += 1
                byte_cnt += 1

            batch_num += 1

            conv_loss.append(packet_loss)
            conv_acc.append(packet_good_cnt / len(cur_packet.payload))

        context.append(
            model.packet_encoder.forward(
                target_payload,
                cur_packet.cat_features.to(device=DEVICE),
                cur_packet.numerical_features.to(device=DEVICE),
            ).detach()
        )
        context = context[-H_PACKET_CTX:]

    if show_plots:
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
        correct_byte_cnt / byte_cnt if byte_cnt > 0 else float("inf"),
    )


def train_epoch(
    conv_dfs: List[PacketDataset],
    model: HeirarchicalMQTTModel,
    optimizer,
    show_plots: bool = DEBUG_MODE,
) -> EpochResults:
    """
    @Description: Runs the entier set of conversation data frames through the training and
    validation process.

    @Notes:
        - The training and validaton is done at the packet level

    @Returns:
    """
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
    global g_cur_temperature

    # Metrics
    best_val_loss = float("inf")
    train_losses = list()
    val_losses = list()
    train_accs = list()
    val_accs = list()

    # Declare empty model, optimizer and criterion
    heirachical_lstm = None
    optimizer = None
    scheduler = None

    # Now train over n training epochs
    for epoch in range(N_EPOCHS):
        g_cur_temperature = P_INITIAL_TEMPERATURE * (1 - epoch / N_EPOCHS)
        dfs = load_dfs_from_dir(csv_dir=csv_dir)
        for df in dfs:
            # Get the conversations splits
            splits = split_into_conversations(df, conv_list=conv_list)

            conv_dfs = [
                PacketDataset(conv_df, n_convs=len(conv_list)) for conv_df in splits
            ]

            # Since we now have the features and dimensions we can initialize the model
            if heirachical_lstm is None:
                cat_dims = conv_dfs[0].cat_dims
                num_dims = conv_dfs[0].num_dims
                heirachical_lstm = HeirarchicalMQTTModel(
                    categorical_dims=cat_dims, numerical_dim=num_dims, device=DEVICE
                )
                optimizer = torch.optim.Adam(
                    heirachical_lstm.parameters(),
                    lr=P_LEARNING_RATE,
                    weight_decay=P_WEIGHT_DECAY,
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer, mode="min", factor=0.1, patience=10
                )

            results = train_epoch(
                conv_dfs, heirachical_lstm, optimizer=optimizer, show_plots=False
            )

            print(f"Epoch {epoch+1}/{N_EPOCHS}:")
            print(f"  Training Loss: {results.avg_train_loss:.4f}")
            print(f"  Training Acc: {results.avg_train_acc:.4f}")

            train_losses.append(results.avg_train_loss)
            train_accs.append(results.avg_train_acc)
            val_losses.append(results.avg_val_loss)
            val_accs.append(results.avg_val_acc)

            # Update the training parameters
            scheduler.step(results.avg_val_loss)

            # Model checkpointing
            if results.avg_val_loss < best_val_loss:
                best_val_loss = results.avg_val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": heirachical_lstm.state_dict(),
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


if __name__ == "__main__":
    ### Training entry point ###
    # csv_dir = "datasets/mqtt-data/kaggle_mqtt_set/Data/PCAP/legit_cap_split/legtimate_w1-1_split"
    csv_dir = "test_data"
    # csv_dir = (
    #     "datasets/mqtt-data/kaggle_mqtt_set/Data/PCAP/legit_cap_split/small_sample"
    # )
    model_train(csv_dir=csv_dir)
