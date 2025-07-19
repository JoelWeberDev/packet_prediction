import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional

class ByteSequenceDataset(Dataset):
    """Dataset for byte sequences with categorical features"""
    
    def __init__(self, sequences: List[bytes], categorical_features: List[List[int]], 
                 sequence_length: int = 128):
        """
        Args:
            sequences: List of byte sequences
            categorical_features: List of categorical feature arrays (aligned with sequences)
            sequence_length: Length of input sequences for training
        """
        self.sequences = sequences
        self.categorical_features = categorical_features
        self.sequence_length = sequence_length
        self.samples = []
        
        # Create training samples
        self._create_samples()
    
    def _create_samples(self):
        """Create input/target pairs from sequences"""
        for seq_idx, (sequence, cat_features) in enumerate(zip(self.sequences, self.categorical_features)):
            # Convert bytes to list of integers
            byte_sequence = list(sequence)
            
            # Create sliding windows
            for i in range(len(byte_sequence) - self.sequence_length):
                input_bytes = byte_sequence[i:i + self.sequence_length]
                target_byte = byte_sequence[i + self.sequence_length]
                
                # Get corresponding categorical features
                input_cat_features = cat_features[i:i + self.sequence_length]
                
                self.samples.append({
                    'input_bytes': torch.tensor(input_bytes, dtype=torch.long),
                    'input_categorical': torch.tensor(input_cat_features, dtype=torch.long),
                    'target': torch.tensor(target_byte, dtype=torch.long)
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class MultimodalLSTM(nn.Module):
    """LSTM model for next byte prediction with categorical features"""
    
    def __init__(self, categorical_cardinalities: List[int], 
                 byte_embed_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            categorical_cardinalities: List of number of categories for each categorical feature
            byte_embed_dim: Dimension of byte embeddings
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(MultimodalLSTM, self).__init__()
        
        # Byte embedding (256 possible byte values)
        self.byte_embedding = nn.Embedding(256, byte_embed_dim)
        
        # Categorical embeddings
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_cats, min(50, (num_cats + 1) // 2))
            for num_cats in categorical_cardinalities
        ])
        
        # Calculate total input dimension
        cat_embed_dims = [min(50, (num_cats + 1) // 2) for num_cats in categorical_cardinalities]
        self.total_input_dim = byte_embed_dim + sum(cat_embed_dims)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.total_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer (project to 256 possible bytes)
        self.output_layer = nn.Linear(hidden_dim, 256)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, byte_sequences, categorical_sequences):
        """
        Args:
            byte_sequences: [batch_size, seq_len]
            categorical_sequences: [batch_size, seq_len, num_categorical_features]
        
        Returns:
            logits: [batch_size, 256] - logits for next byte prediction
        """
        batch_size, seq_len = byte_sequences.shape
        
        # Embed bytes
        byte_embeds = self.byte_embedding(byte_sequences)  # [batch, seq_len, byte_embed_dim]
        
        # Embed categorical features
        cat_embeds = []
        for i, embedding_layer in enumerate(self.categorical_embeddings):
            cat_embed = embedding_layer(categorical_sequences[:, :, i])  # [batch, seq_len, cat_embed_dim]
            cat_embeds.append(cat_embed)
        
        # Concatenate all embeddings
        if cat_embeds:
            all_embeds = torch.cat([byte_embeds] + cat_embeds, dim=-1)  # [batch, seq_len, total_input_dim]
        else:
            all_embeds = byte_embeds
        
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(all_embeds)  # lstm_out: [batch, seq_len, hidden_dim]
        
        # Use the last timestep's output for prediction
        final_hidden = lstm_out[:, -1, :]  # [batch, hidden_dim]
        final_hidden = self.dropout(final_hidden)
        
        # Project to vocabulary size
        logits = self.output_layer(final_hidden)  # [batch, 256]
        
        return logits


def train_model(model, train_loader, val_loader, num_epochs, device, 
                learning_rate=0.001, clip_grad_norm=1.0):
    """Training loop for the LSTM model"""
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            input_bytes = batch['input_bytes'].to(device)
            input_categorical = batch['input_categorical'].to(device)
            targets = batch['target'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_bytes, input_categorical)
            loss = criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            # Update weights
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*train_correct/train_total:.2f}%')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_bytes = batch['input_bytes'].to(device)
                input_categorical = batch['input_categorical'].to(device)
                targets = batch['target'].to(device)
                
                logits = model(input_bytes, input_categorical)
                loss = criterion(logits, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
        
        # Learning rate scheduling
        scheduler.step(val_loss)


def generate_text(model, initial_sequence, categorical_features, 
                  max_length=100, temperature=1.0, device='cpu'):
    """Generate text using the trained model"""
    model.eval()
    model.to(device)
    
    generated = list(initial_sequence)
    current_cat_features = categorical_features.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input (use last sequence_length bytes)
            seq_len = model.lstm.input_size  # This won't work - need to store sequence length
            input_bytes = torch.tensor(generated[-seq_len:], dtype=torch.long).unsqueeze(0).to(device)
            input_cat = torch.tensor(current_cat_features[-seq_len:], dtype=torch.long).unsqueeze(0).to(device)
            
            # Predict next byte
            logits = model(input_bytes, input_cat)
            
            # Apply temperature
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1)
            
            # Sample next byte
            next_byte = torch.multinomial(probabilities, 1).item()
            generated.append(next_byte)
            
            # Update categorical features (you'd need to implement this based on your features)
            # current_cat_features.append(get_next_categorical_features(next_byte))
    
    return bytes(generated)


# Example usage
if __name__ == "__main__":
    # Example data preparation
    sequences = [b"Hello world", b"Machine learning", b"Neural networks"]  # Your byte sequences
    categorical_features = [
        [[0, 1, 2], [1, 2, 0]],  # Example categorical features for each sequence
        [[2, 0, 1], [0, 1, 2]],
        [[1, 2, 0], [2, 0, 1]]
    ]
    
    # Model parameters
    categorical_cardinalities = [7, 4, 3]  # Number of categories for each feature
    
    # Create dataset and dataloader
    dataset = ByteSequenceDataset(sequences, categorical_features, sequence_length=32)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = MultimodalLSTM(categorical_cardinalities)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, train_loader, val_loader, num_epochs=50, device=device)
    
    # Save model
    torch.save(model.state_dict(), 'lstm_byte_model.pth')