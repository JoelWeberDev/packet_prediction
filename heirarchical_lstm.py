import torch
import torch.nn as nn

class PacketEncoder(nn.Module):
    """Encodes individual packet (metadata + payload) into fixed representation"""
    
    def __init__(self, byte_vocab_size=256, byte_embed_dim=128, 
                 categorical_dims=None, numerical_dim=0, packet_repr_dim=256):
        super().__init__()
        
        # Byte embedding
        self.byte_embedding = nn.Embedding(byte_vocab_size, byte_embed_dim)
        
        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_size, max(50, cat_size // 2)) 
            for cat_size in categorical_dims
        ])
        
        # Calculate input dimension for packet LSTM
        cat_embed_dim = sum(max(50, cat_size // 2) for cat_size in categorical_dims)
        packet_lstm_input_dim = byte_embed_dim + cat_embed_dim + numerical_dim
        
        # Packet-level LSTM (processes bytes within packet)
        self.packet_lstm = nn.LSTM(
            input_size=packet_lstm_input_dim,
            hidden_size=packet_repr_dim // 2,  # Will be bidirectional
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Additional MLP to process metadata separately
        self.metadata_mlp = nn.Sequential(
            nn.Linear(cat_embed_dim + numerical_dim, packet_repr_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Combine packet content + metadata
        self.packet_combiner = nn.Sequential(
            nn.Linear(packet_repr_dim + packet_repr_dim // 4, packet_repr_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, byte_sequence, categorical_features, numerical_features):
        batch_size, seq_len = byte_sequence.shape
        
        # Embed bytes
        byte_embeds = self.byte_embedding(byte_sequence)  # [batch, seq_len, embed_dim]
        
        # Embed categorical features and repeat for sequence
        cat_embeds = []
        for i, embedding_layer in enumerate(self.cat_embeddings):
            cat_embed = embedding_layer(categorical_features[:, i])
            cat_embed = cat_embed.unsqueeze(1).repeat(1, seq_len, 1)
            cat_embeds.append(cat_embed)
        cat_embeds = torch.cat(cat_embeds, dim=-1) if cat_embeds else torch.empty(batch_size, seq_len, 0)
        
        # Repeat numerical features for sequence
        num_embeds = numerical_features.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Combine all inputs
        packet_input = torch.cat([byte_embeds, cat_embeds, num_embeds], dim=-1)
        
        # Process through packet LSTM
        lstm_output, _ = self.packet_lstm(packet_input)
        
        # Take final hidden state as packet representation
        packet_content_repr = lstm_output[:, -1, :]  # [batch, packet_repr_dim]
        
        # Process metadata separately
        metadata_input = torch.cat([cat_embeds[:, 0, :], numerical_features], dim=-1)
        metadata_repr = self.metadata_mlp(metadata_input)
        
        # Combine representations
        packet_repr = self.packet_combiner(torch.cat([packet_content_repr, metadata_repr], dim=-1))
        
        return packet_repr


class ConversationLSTM(nn.Module):
    """Maintains conversation state across packets"""
    
    def __init__(self, packet_repr_dim=256, conversation_hidden_dim=512, num_layers=3):
        super().__init__()
        
        self.conversation_lstm = nn.LSTM(
            input_size=packet_repr_dim,
            hidden_size=conversation_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        self.conversation_hidden_dim = conversation_hidden_dim
        self.num_layers = num_layers
    
    def forward(self, packet_representations, hidden_state=None):
        """
        Args:
            packet_representations: [batch, num_packets, packet_repr_dim]
            hidden_state: Previous conversation state (h, c)
        Returns:
            conversation_outputs: [batch, num_packets, conversation_hidden_dim]
            final_hidden_state: Updated conversation state
        """
        conversation_outputs, final_hidden_state = self.conversation_lstm(
            packet_representations, hidden_state
        )
        return conversation_outputs, final_hidden_state
    
    def init_hidden(self, batch_size, device):
        """Initialize conversation hidden state"""
        h = torch.zeros(self.num_layers, batch_size, self.conversation_hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.conversation_hidden_dim, device=device)
        return (h, c)


class NextPacketPredictor(nn.Module):
    """Predicts next packet payload given conversation context"""
    
    def __init__(self, conversation_hidden_dim=512, categorical_dims=None, 
                 numerical_dim=0, max_payload_length=128, byte_vocab_size=256):
        super().__init__()
        
        # Encode next packet metadata
        cat_embed_dim = sum(max(50, cat_size // 2) for cat_size in categorical_dims)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_size, max(50, cat_size // 2)) 
            for cat_size in categorical_dims
        ])
        
        # Combine conversation context with next packet metadata
        context_dim = conversation_hidden_dim + cat_embed_dim + numerical_dim
        
        # Decoder LSTM for payload generation
        self.decoder_lstm = nn.LSTM(
            input_size=context_dim + 128,  # context + previous byte embedding
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Output projection
        self.output_projection = nn.Linear(256, byte_vocab_size)
        
        # Byte embedding for decoder
        self.byte_embedding = nn.Embedding(byte_vocab_size, 128)
        
        self.max_payload_length = max_payload_length
    
    def forward(self, conversation_context, next_packet_categorical, 
                next_packet_numerical, target_payload=None):
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
        cat_embeds = torch.cat(cat_embeds, dim=-1) if cat_embeds else torch.empty(batch_size, 0)
        
        # Combine context with next packet metadata
        context = torch.cat([conversation_context, cat_embeds, next_packet_numerical], dim=-1)
        
        if target_payload is not None:
            # Training mode - teacher forcing
            return self._train_forward(context, target_payload)
        else:
            # Inference mode - autoregressive generation
            return self._generate_payload(context)
    
    def _train_forward(self, context, target_payload):
        batch_size, payload_length = target_payload.shape
        
        # Prepare decoder input (start with <SOS> token, assume byte 0)
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=target_payload.device)
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
    
    def __init__(self, categorical_dims, numerical_dim):
        super().__init__()
        
        self.packet_encoder = PacketEncoder(
            categorical_dims=categorical_dims,
            numerical_dim=numerical_dim
        )
        
        self.conversation_lstm = ConversationLSTM()
        
        self.next_packet_predictor = NextPacketPredictor(
            categorical_dims=categorical_dims,
            numerical_dim=numerical_dim
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


# Training loop would handle conversation windowing
# For 10k packets, you'd probably use sliding windows of ~100-500 packets
# and maintain conversation state across windows