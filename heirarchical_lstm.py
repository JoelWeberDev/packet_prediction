import torch
import torch.nn as nn
from typing import List

# Local imports
from CONSTANTS import *
from preprocessing import load_df, split_into_conversations
from custom_datasets import PacketDataset


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

        # Embed bytes
        byte_embeds = self.byte_embedding(byte_sequence)  # [batch, seq_len, embed_dim]

        mask = attention_mask.unsqueeze(-1)
        byte_embeds = (
            byte_embeds * mask
        )  # zeros any out of range parameters in the byte embeddings

        # Embed categorical features and repeat for sequence
        cat_embeds = []
        for i, embedding_layer in enumerate(self.cat_embeddings):
            cat_embed = embedding_layer(categorical_features[:, i])
            cat_embeds.append(cat_embed)
        cat_embeds = (
            torch.cat(cat_embeds, dim=-1) if cat_embeds else torch.empty(batch_size, 0)
        )

        # Combine all inputs
        # packet_input = torch.cat(
        #     [byte_embeds, cat_embeds.unsqueeze(1), numerical_features.unsqueeze(1)],
        #     dim=-1,
        # )
        packet_input = byte_embeds

        # Now pack the embeddings using the attention mask
        lengths = attention_mask.sum(dim=1).cpu()  # [batch_size]
        packed_embeds = nn.utils.rnn.pack_padded_sequence(
            packet_input, lengths, batch_first=True, enforce_sorted=False
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
        conversation_context,
        next_packet_categorical,
        next_packet_numerical,
        target_payload=None,
    ):
        """
        Args:
            conversation_context: [batch, conversation_hidden_dim]
            next_packet_categorical: [batch, num_categorical]
            next_packet_numerical: [batch, numerical_dim]
            target_payload: [batch, payload_length] - for teacher forcing during training
        """
        batch_size = conversation_context.shape[0]

        # Embed next packet categorical features
        cat_embeds = []
        for i, embedding_layer in enumerate(self.cat_embeddings):
            cat_embeds.append(embedding_layer(next_packet_categorical[:, i]))
        cat_embeds = (
            torch.cat(cat_embeds, dim=-1) if cat_embeds else torch.empty(batch_size, 0)
        )

        # Combine context with next packet metadata
        context = torch.cat(
            [conversation_context, cat_embeds, next_packet_numerical], dim=-1
        )

        if target_payload is not None:
            # Training mode - teacher forcing
            return self._train_forward(context, target_payload)
        else:
            # Inference mode - autoregressive generation
            return self._generate_payload(context)

    def _train_forward(self, context, target_payload):
        batch_size, payload_length = target_payload.shape

        # Prepare decoder input (start with <SOS> token, assume byte 0)
        decoder_input = torch.zeros(
            batch_size, 1, dtype=torch.long, device=target_payload.device
        )
        decoder_input = torch.cat([decoder_input, target_payload[:, :-1]], dim=1)

        # Embed decoder input
        decoder_embeds = self.byte_embedding(decoder_input)

        # Repeat context for each timestep
        context_repeated = context.unsqueeze(1).repeat(1, payload_length, 1)

        # Combine context with decoder embeddings
        lstm_input = torch.cat([context_repeated, decoder_embeds], dim=-1)

        # Process through decoder LSTM
        decoder_output, _ = self.decoder_lstm(lstm_input)

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits

    def _generate_payload(self, context):
        # Implement autoregressive generation for inference
        # This would generate bytes one by one using the trained model
        pass


class HierarchicalMQTTModel(nn.Module):
    """Complete hierarchical model for MQTT conversation modeling"""

    def __init__(self, categorical_dims: list | tuple, numerical_dim: int):
        super().__init__()

        self.packet_encoder = PacketEncoder(
            categorical_dims=categorical_dims, numerical_dim=numerical_dim
        )

        self.conversation_lstm = ConversationLSTM()

        self.next_packet_predictor = NextPacketPredictor(
            categorical_dims=categorical_dims, numerical_dim=numerical_dim
        )

    def forward(self, conversation_packets, next_packet_metadata, target_payload=None):
        """
        Args:
            conversation_packets: List of (bytes, categorical, numerical) for each packet in conversation
            next_packet_metadata: (categorical, numerical) for packet we're predicting
            target_payload: Target payload for training
        """
        # Encode all packets in conversation
        packet_representations = []
        for packet_bytes, packet_cat, packet_num in conversation_packets:
            packet_repr = self.packet_encoder(packet_bytes, packet_cat, packet_num)
            packet_representations.append(packet_repr)

        packet_representations = torch.stack(packet_representations, dim=1)

        # Process through conversation LSTM
        conversation_outputs, _ = self.conversation_lstm(packet_representations)

        # Use final conversation state to predict next packet
        final_conversation_context = conversation_outputs[:, -1, :]

        # Predict next packet payload
        next_cat, next_num = next_packet_metadata
        logits = self.next_packet_predictor(
            final_conversation_context, next_cat, next_num, target_payload
        )

        return logits


if __name__ == "__main__":
    ### Training section ###
    def evaluate(
        mqtt_model: HierarchicalMQTTModel, conv_data: List[PacketDataset]
    ) -> float:
        return 0.0

    def train_epoch(
        mqtt_mode: HierarchicalMQTTModel, conv_data: List[PacketDataset]
    ) -> float:
        return 0.0

    def save_checkpoint(mqtt_model: HierarchicalMQTTModel):
        pass

    def conversation_train(
        mqtt_model: HierarchicalMQTTModel, convs: List[PacketDataset]
    ):
        """
        @Description: Splits the dataset into training testing and validation splits
        We then train and test the model with this.

        @Notes:
            - We use the training / validation / testing splits defined in constants
            for our split percentages

        @Returns:
        """
        train_size = int(TRAIN_VAL_TEST_PERC[0] * len(convs))
        val_size = int(TRAIN_VAL_TEST_PERC[1] * len(convs))

        assert val_size > 0, f"Cannot train with testing and validation sizes of 0"

        train_convs = convs[:train_size]
        val_convs = convs[train_size : train_size + val_size]
        test_convs = convs[train_size + val_size :]

        best_val_loss = 0

        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch(mqtt_model, train_convs)

            val_loss = evaluate(mqtt_model, val_convs)

            # Early stopping criteria
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(mqtt_model)

        test_loss = evaluate(mqtt_model, test_convs)
        print(f"Final Test Loss: {test_loss:.4f}")

    def model_train():
        # Get the dataset for the conversation data
        df = load_df()

        # Split the data set into conversations by conversation number
        conv_dfs = [PacketDataset(conv_df) for conv_df in split_into_conversations(df)]

        if len(conv_dfs) == 0:
            return

        # Get the categorical and numerical dimensions. These are all identical throughout the datasets
        # Define the cross entropy loss model and optimizer
        mqtt_model = HierarchicalMQTTModel(
            categorical_dims=conv_dfs[0].cat_dims,
            numerical_dim=conv_dfs[0].numerical_dims,
        )

        # preform the testing here

        # Establish the number of epochs and create training splits for that
        pass
