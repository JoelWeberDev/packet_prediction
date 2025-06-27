"""
@Author: Joel Weber
@Date: 2025-06-20
@Description:

@Notes:
    Workflow design:
        - For a training step we need a context and target packet
        - Is passed to the hiearchical model where it creates context embeddings with the
        target meta data and historical packets.
            - These historical packets are embedded using a packet embedded that takes the
            full payload along with the features for each packet.
        - This context along with the target payload is sent to the next packet predictor which
            - The next packet metadata is concatenated to the context embeddings
            - Repeats the context for each unmasked byte in the target payload
            - Creates a logits byte prediciton for each of the unmasked bytes in the sequence.

@Questions:
    - Is the next byte predictor going to work in the training phase
    - How can we test where the pipeline is failing?
    - How are batch sizes going to be done?


@TODO:
    - Implement autoregresive next packet prediction
    - Pad all the target sequences to length and construct an attention mask for them
    - Training
        - Data preprocessing
        - Define the optimizer
        - Create a cross entropy loss function for the optimization

"""

import sys, os
import torch
import torch.nn as nn
from typing import List, Dict
from dataclasses import dataclass

# Local imports
from CONSTANTS import *
from preprocessing import load_df, split_into_conversations
from custom_datasets import PacketDataset, SeqInput, ParsedPacket, PacketWithContext


class PacketEncoder(nn.Module):
    """Encodes individual packet (metadata + payload) into fixed representation"""

    def __init__(
        self,
        categorical_dims=list(),
        numerical_dim=0,
    ):
        super().__init__()

        # Byte embedding
        self.byte_embedding = nn.Embedding(BYTE_VOCAB_DIM, BYTE_EMBED_DIM)

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(cat_size, max(50, cat_size // 2))
                for cat_size in categorical_dims
            ]
        )

        # Calculate input dimension for packet LSTM
        cat_embed_dim = sum(max(50, cat_size // 2) for cat_size in categorical_dims)
        # packet_lstm_input_dim = byte_embed_dim + cat_embed_dim + numerical_dim
        packet_lstm_input_dim = BYTE_EMBED_DIM

        # Packet-level LSTM (processes bytes within packet)
        self.packet_lstm = nn.LSTM(
            input_size=packet_lstm_input_dim,
            hidden_size=PACKET_REP_DIM // 2,  # Will be bidirectional
            num_layers=PACKET_ENC_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=PACKET_ENC_DROPOUT,
        )

        # Additional MLP to process metadata separately
        self.metadata_mlp = nn.Sequential(
            nn.Linear(cat_embed_dim + numerical_dim, PACKET_REP_DIM // 4),
            nn.ReLU(),
            nn.Dropout(METADATA_MLP_DROPOUT),
        )

        # Combine packet content + metadata
        self.packet_combiner = nn.Sequential(
            nn.Linear(PACKET_REP_DIM + PACKET_REP_DIM // 4, PACKET_REP_DIM),
            nn.ReLU(),
            nn.Dropout(PACKET_COMBINER_DROPOUT),
        )

    def forward(
        self,
        byte_sequence,
        attention_mask,
        categorical_features,
        numerical_features,
    ):
        """
        @Description: Forward pass for the single packet embedding and prediciton LSTM model.
        This takes a batch of inputs in the byte sequence, and produces the output predictions
        for that input. Here we also are given the corresponding attention masks for the
        each byte sequence. The attention mask must have the same shape as the byte sequence.

        @Notes:
            - The numerical and categorical features remain constant throughout the packet
            - The batch size may vary depending on the length of the sequence. Typically
            we cap the batch size at min(32, sequnce length)

        @Returns:
        """
        batch_size, seq_len = byte_sequence.shape

        # Embed categorical features and repeat for sequence
        cat_embeds = []
        for i, embedding_layer in enumerate(self.cat_embeddings):
            cat_embed = embedding_layer(categorical_features[:, i])
            cat_embeds.append(cat_embed)
        cat_embeds = (
            torch.cat(cat_embeds, dim=-1) if cat_embeds else torch.empty(batch_size, 0)
        )

        # Embed bytes
        byte_embeds = self.byte_embedding(byte_sequence)  # [batch, seq_len, embed_dim]

        # Now pack the embeddings using the attention mask
        lengths = attention_mask.sum(dim=1).cpu()  # [batch_size]
        packed_embeds = nn.utils.rnn.pack_padded_sequence(
            byte_embeds, lengths, batch_first=True, enforce_sorted=False
        )

        # Process through packet LSTM
        lstm_output, _ = self.packet_lstm(packed_embeds)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)

        # Take final hidden state as packet representation
        packet_content_repr = lstm_output[:, -1, :]  # [batch, packet_repr_dim]

        # Process metadata separately
        metadata_input = torch.cat([cat_embeds, numerical_features], dim=-1)
        metadata_repr = self.metadata_mlp(metadata_input)

        # Combine representations
        packet_repr = self.packet_combiner(
            torch.cat([packet_content_repr, metadata_repr], dim=-1)
        )

        return packet_repr


class ConversationLSTM(nn.Module):
    """Maintains conversation state across packets"""

    def __init__(self):
        super().__init__()

        self.conversation_lstm = nn.LSTM(
            input_size=PACKET_REP_DIM,
            hidden_size=CONVERSATIONAL_HIDDEN_DIM,
            num_layers=CONVERSATIONAL_LAYERS,
            batch_first=True,
            dropout=CONV_LSTM_DROPOUT,
        )

    def forward(self, packet_representations, hidden_state=None):
        """
        @Args:
            packet_representations: [batch, num_packets, packet_repr_dim]
            hidden_state: Previous conversation state (h, c)
        @Returns:
            conversation_outputs: [batch, num_packets, conversation_hidden_dim]
            final_hidden_state: Updated conversation state
        """
        conversation_outputs, final_hidden_state = self.conversation_lstm(
            packet_representations, hidden_state
        )
        return conversation_outputs, final_hidden_state

    def init_hidden(self, batch_size, device):
        """Initialize conversation hidden state"""
        h = torch.zeros(
            CONVERSATIONAL_LAYERS, batch_size, CONVERSATIONAL_HIDDEN_DIM, device=device
        )
        c = torch.zeros(
            CONVERSATIONAL_LAYERS, batch_size, CONVERSATIONAL_HIDDEN_DIM, device=device
        )
        return (h, c)


class NextPacketPredictor(nn.Module):
    """Predicts next packet payload given conversation context"""

    def __init__(self, categorical_dims=list(), numerical_dim=0):
        super().__init__()

        # Encode next packet metadata
        cat_embed_dim = sum(max(50, cat_size // 2) for cat_size in categorical_dims)
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(cat_size, max(50, cat_size // 2))
                for cat_size in categorical_dims
            ]
        )

        # Combine conversation context with next packet metadata
        context_dim = CONVERSATIONAL_HIDDEN_DIM + cat_embed_dim + numerical_dim

        # Decoder LSTM for payload generation
        self.decoder_lstm = nn.LSTM(
            input_size=context_dim
            + BYTE_EMBED_DIM,  # context + previous byte embedding
            hidden_size=CONVERSATIONAL_HIDDEN_DIM,
            num_layers=NEXT_PACKET_LAYERS,
            batch_first=True,
            dropout=NEXT_PACKET_DROPOUT,
        )

        # Output projection
        self.output_projection = nn.Linear(CONVERSATIONAL_HIDDEN_DIM, BYTE_VOCAB_DIM)

        # Byte embedding for decoder
        self.byte_embedding = nn.Embedding(BYTE_VOCAB_DIM, BYTE_EMBED_DIM)

    def forward(
        self,
        embedded_conversation_context: torch.Tensor,  # embedded context
        next_packet_categorical: torch.Tensor,
        next_packet_numerical: torch.Tensor,
        target_payload: None | torch.Tensor = None,
        attn_mask: None | torch.Tensor = None,
    ):
        """
        @Args:
            conversation_context: [batch, conversation_hidden_dim]
            next_packet_categorical: [batch, num_categorical]
            next_packet_numerical: [batch, numerical_dim]
            target_payload: [batch, payload_length] - for teacher forcing during training
        @Notes:
            - The mqtt.len is the third arguement in the numerical features
        """
        batch_size = embedded_conversation_context.shape[0]

        # Embed next packet categorical features
        cat_embeds = []
        for i, embedding_layer in enumerate(self.cat_embeddings):
            cat_embeds.append(embedding_layer(next_packet_categorical[:, i]))

        cat_embeds = (
            torch.cat(cat_embeds, dim=-1) if cat_embeds else torch.empty(batch_size, 0)
        )

        # Combine context with next packet metadata as predictors
        context = torch.cat(
            [embedded_conversation_context, cat_embeds, next_packet_numerical], dim=-1
        )

        if target_payload is None:
            # Get the mqtt length as the third argument in the numerical features
            lengths = next_packet_numerical[:, 2]
            # Inference mode - autoregressive generation
            return self._generate_payload(context, lengths)
        else:
            assert not isinstance(
                attn_mask, type(None)
            ), f"An attention mask corresponding to the target payload must be provided"

            assert (
                target_payload.shape == attn_mask.shape
            ), f"Target payload shape: {target_payload.shape} != attention mask shape: {attn_mask.shape}"

            # Training mode - teacher forcing
            return self._train_forward(context, target_payload, attn_mask)

    def _train_forward(
        self, context, target_payload: torch.Tensor, attn_mask: torch.Tensor
    ):
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
            logits: torch.tensor([]) -> [batch_size, payload_length, BYTE_VOCAB_DIMS]
        """
        batch_size, max_len = target_payload.shape  # [batch_size, payload_length]

        # Since we are dealing with variable payload lengths we will need to pack the payloads
        # This operates on the assumption that
        payload_lens = attn_mask.sum(dim=1)

        # Embed decoder input
        decoder_embeds = self.byte_embedding(
            target_payload
        )  # [batch_size, max_len, BYTE_EMBED_DIMS]

        # ensure that the attention mask is applied to the embeddings
        mask = attn_mask.unsqueeze(-1)
        mask_embeds = decoder_embeds * mask

        # Repeat context for each timestep in the payload so that we predict byte by byte
        context_repeated = context.unsqueeze(1).repeat(1, max_len, 1)

        # Combine context with decoder embeddings
        lstm_input = torch.cat([context_repeated, mask_embeds], dim=-1)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            lstm_input, payload_lens.cpu(), batch_first=True, enforce_sorted=False
        )

        # Process through decoder LSTM and unpad
        packed_output, _ = self.decoder_lstm(packed_input)
        decoder_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits  # [batch_size, payload_size, BYTE_EMBED_DIMS]

    def _generate_payload(
        self, context: torch.Tensor, msg_lens: torch.Tensor
    ) -> List[SeqInput]:
        """
        @Description: Goes through autoregressively predicting the packet payload byte by byte

        @Notes:
            - The length of the payload must be provided. Typically this is contained within
            the meta data.

        @TEST:
            - Does this padded sequnce only predict for a single byte? What is the output shape?

        @Returns: List[SeqInput]
        """
        # Autoregressively predict the next byte in the packet, adding it to the context each time
        ctx_batch_size, context_len = context.shape
        batch_size = msg_lens.shape[0]
        assert (
            ctx_batch_size == batch_size
        ), f"_generate_payload failed with batch size mismatch {batch_size} != {ctx_batch_size}"

        max_len = int(msg_lens.max())
        pred_payloads = torch.zeros((batch_size, max_len), dtype=torch.uint8)

        for i in range(max_len):
            # Encode only the prior bytes
            byte_embeds = self.byte_embedding(pred_payloads[:, :i])

            # make a copy of the context for the current timestamp
            context_step = context.unsqueeze(1)

            if i > 1:
                lstm_input = torch.cat([context_step, byte_embeds], dim=1)
            else:
                lstm_input = context_step

            # Get the LSTM output for the given input
            output, _ = self.decoder_lstm(lstm_input)

            logits = self.output_projection(
                output[:, -1, :]
            )  # Only gets the very last prediction
            # Select the most likely byte for each
            pred_byte = logits.argmax(dim=-1)

            pred_payloads[:, i] = pred_byte

        # Now create attention mask for each msg length
        attn_masks = torch.tensor(
            [[1] * s_len + [0] * (max_len - s_len) for s_len in msg_lens],
            dtype=torch.bool,
        )

        # Now construct the sequence of SeqInputs
        return [
            SeqInput(payload, attn_mask, int(attn_mask.sum(dim=1)))
            for payload, attn_mask in zip(pred_payloads, attn_masks)
        ]


class HierarchicalMQTTModel(nn.Module):
    """Complete hierarchical model for MQTT conversation modeling"""

    def __init__(
        self, categorical_dims: list | tuple, numerical_dim: int, device: str = DEVICE
    ):
        super().__init__()

        self.packet_encoder = PacketEncoder(
            categorical_dims=categorical_dims, numerical_dim=numerical_dim
        )

        self.conversation_lstm = ConversationLSTM()

        self.next_packet_predictor = NextPacketPredictor(
            categorical_dims=categorical_dims, numerical_dim=numerical_dim
        )

        self.device = device
        self.to(device)

    def forward(
        self,
        conversation_packets: List[List[ParsedPacket]],
        next_cat: torch.Tensor,
        next_numerical: torch.Tensor,
        target_payload: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
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
        batch_size = len(conversation_packets)
        context_length = len(conversation_packets[0])

        for ctx in conversation_packets:
            assert (
                len(ctx) == context_length
            ), f"All the packets must have the same context length of {context_length}"

        # Embedded packet shape: [byte_embeddings + cat_embed_dim + num_embed_dim]
        # individual embedding shape: [context_length, embedded_packet_length]
        packet_representations = list()
        for i in range(context_length):
            # get the context packets
            # packet = conversation_packets[:, i]  # [batch_size, 1]
            packets = [ctx[i] for ctx in conversation_packets]

            # extract the individual elements from each packets
            payload = torch.stack([p.padded_payload.input_bytes for p in packets]).to(
                DEVICE
            )
            attn_mask = torch.stack(
                [p.padded_payload.attention_mask for p in packets]
            ).to(DEVICE)
            cat_feats = torch.stack([p.cat_features for p in packets]).to(DEVICE)
            numerical_feats = torch.stack([p.numerical_features for p in packets]).to(
                DEVICE
            )

            # TODO: Test how this is encoded
            packet_repr = self.packet_encoder(
                payload, attn_mask, cat_feats, numerical_feats
            )
            packet_representations.append(packet_repr)

        packet_representations = torch.stack(packet_representations, dim=1).to(DEVICE)

        # Process through conversation LSTM
        conversation_outputs, _ = self.conversation_lstm(packet_representations)

        # Use final conversation state to predict next packet
        final_conversation_context = conversation_outputs[:, -1, :]

        # Predict next packet payload
        logits = self.next_packet_predictor(
            final_conversation_context,
            next_cat,
            next_numerical,
            target_payload,
            attn_mask,
        )

        return logits


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


def training_step(model, batch: List[PacketWithContext], optimizer, criterion) -> float:
    """
    @Description: Each training step works to predict the next packet in the conversation.
    To do this it requires a context of length CONV_CONTEXT_LEN and a target packet; the
    next packet in the sequence.

    @Notes:
        - We are aiming to back prop a single gradient that applies to all the weights and
        parameters of the LSTM model rather than performing and partitioned optimization.
        The hierchical mqtt model is the front end for the rest of the model. Whatever we pass
        to there we can use to optimize the entier model.
        - A single step forward requires a context of past packets, next categorical and numerical
        features and the target payload.
        - When training we use a batch size defined in the constants file. In our batch we go
        through incrementally packet by packet where the next row in a batch will have the last
        target packet in its context.

    @Returns:
    """
    # Parse the data into the proper format
    conversation_packets = [pc.context for pc in batch]
    next_cat = torch.stack([pd.target.cat_features for pd in batch]).to(
        model.device
    )  # [batch_size, cat_len]
    next_numerical = torch.stack([pd.target.numerical_features for pd in batch]).to(
        model.device
    )  # [batch_size, num_len]
    # Now use the attention masks to ignore any predictions out of range
    target_payload = torch.stack(
        [pd.target.padded_payload.input_bytes for pd in batch]
    ).to(model.device)

    attn_mask = torch.stack(
        [pd.target.padded_payload.attention_mask for pd in batch]
    ).to(
        model.device
    )  # [batch_size, max_seq_len]

    optimizer.zero_grad()

    # Reshape the logits to line up all the predictions across the batches
    logits = model(
        conversation_packets, next_cat, next_numerical, target_payload, attn_mask
    )
    # Since we are packing the sequences this reduces the sequence length
    # to the largest attention mask length
    batch_size, max_seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)

    targets = target_payload[:, :max_seq_len].reshape(-1)
    valid_inds = attn_mask[:, :max_seq_len].reshape(-1).bool()

    valid_logits = logits[valid_inds]
    valid_targets = targets[valid_inds]

    # Pass into the criterion
    loss = criterion(valid_logits, valid_targets)

    # Back prop
    loss.backward()
    optimizer.step()

    return loss.item()


def train_conv(model, conv_df: PacketDataset, optimizer, criterion):
    """
    @Description: This does the batching and training for a given conversation

    @Notes:

    @Returns:
    """
    model.train()  # switches to training mode
    total_loss = 0
    batch_count = 0

    # Go through the packets by batch size and perform the training step for each batch
    while True:
        batch = list()
        for _ in range(BATCH_SIZE):
            try:
                batch.append(next(conv_df))
            except StopIteration:
                break

        if len(batch) == 0:
            break

        # Process batch
        loss = training_step(model, batch, optimizer, criterion)
        total_loss += loss
        batch_count += 1

    return total_loss / batch_count if batch_count > 0 else float("inf")


def model_train():
    # Get the dataset for the conversation data
    df = load_df()

    # Each packet dataset represent all the parsed packets in a single conversation
    # Each packet is split into a batch based on the byte sequence for training and comes with
    # some metadata and such as categorical and numerical features.
    conv_dfs = [PacketDataset(df) for df in split_into_conversations(df)]

    if len(conv_dfs) == 0:
        print(f"Number of conversations must not be zero")
        return None, [], []

    # Get the categorical and numerical dimensions they will all be the same throughout the conversations
    cat_dims = conv_dfs[0].cat_dims
    # Adjust the cat dims conversation number to match the total number of conversations
    cat_dims[3] = len(conv_dfs)
    numerical_dim = conv_dfs[0].num_dim

    # Define the cross entropy loss model and optimizer
    mqtt_model = HierarchicalMQTTModel(cat_dims, numerical_dim, device=DEVICE)

    # Now create the optimizer and criterion
    optimizer = torch.optim.Adam(
        mqtt_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
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

        # Set the model in training mode
        mqtt_model.train()
        epoch_loss = 0.0
        for conv_df in train:
            conv_loss = train_conv(mqtt_model, conv_df, optimizer, criterion)
            epoch_loss += conv_loss
        avg_train_loss = epoch_loss / len(conv_dfs)
        train_losses.append(avg_train_loss)

        # now switch to validation
        mqtt_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for conv_df in validation:
                conv_loss = train_conv(mqtt_model, conv_df, optimizer, criterion)
                val_loss += conv_loss
        avg_val_loss = val_loss / len(conv_dfs)
        val_losses.append(avg_val_loss)

        # Model checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": mqtt_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                f"checkpoints/model_epoch_{epoch}.pt",
            )

        # Print metrics
        print(f"Epoch {epoch+1}/{N_EPOCHS}:")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if len(val_losses) > PATIENCE:
            if all(val_losses[-PATIENCE:] > best_val_loss):
                print("Early stopping triggered")
                break

    return mqtt_model, train_losses, val_losses


if __name__ == "__main__":
    # The only thing to do is call the model_train function
    mqtt_model, train_losses, validation_losses = model_train()
