"""
@Author: Joel Weber
@Date: 2025-07-18
@Description: GRU based next packet predictor designed to prefrom extremely well on repetetive
structured data such as conversations in an mqtt network

@Notes:

@TODO:
"""

### Python imports ###
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Iterator
import random
from dataclasses import dataclass

### Local imports ###
from modules.CONSTANTS import *
from modules.preprocessing import load_df, load_dfs_from_dir, split_into_conversations
from modules.custom_datasets import PacketDataset
from modules.helper_functions import (
    split_dict,
    conv_list,
    split_convs,
    print_update,
    plot_metrics,
    hidden_reliance_loss,
    compute_hidden_state_regularization,
    LabelSmoothingCrossEntropy,
    PacketIterator
)


### Custom data structure ###


### Model methods ###
class PacketGenerator(nn.Module):
    """Generate packet payloads based on metadata features"""

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
        )

        # 2. Byte Embeddings
        self.byte_embedding = nn.Embedding(VOCAB_DIM, embedding_size)

        # 3. Decoder Architecture (GRU instead of LSTM - simpler and often better for this task)
        self.decoder = nn.GRU(
            input_size=embedding_size + hidden_size,  # Byte embed + metadata context
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 4. Output projection with proper initialization
        self.output_projection = nn.Linear(
            hidden_size, VOCAB_DIM
        )  # 256 possible byte values
        nn.init.xavier_uniform_(
            self.output_projection.weight, gain=0.1
        )  # Lower gain to prevent overconfidence

        # 5. Mixing Layer
        self.context_mixer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.temperature = 1.0
        self.hidden = None

    def encode_metadata(self, categorical: torch.Tensor, numerical: torch.Tensor):
        """Encode packet metadata into a fixed representation"""
        # Embed categorical features
        cat_embeds = [
            embed(categorical[i]) for i, embed in enumerate(self.cat_embeddings)
        ]
        cat_concat = torch.cat(cat_embeds, dim=0)

        # Combine with numerical features
        metadata_features = torch.cat([cat_concat, numerical], dim=0)

        metadata_enc = self.metadata_encoder(metadata_features)

        # Randomly ablate features during training to prevent reliance on specific patterns
        if self.training:
            # Sometimes completely zero out metadata to force hidden state reliance
            if random.random() < 0.15:  # 15% chance of removing ALL metadata
                metadata_enc = torch.zeros_like(metadata_enc)
            else:
                # Add targeted feature ablation
                feature_mask = torch.bernoulli(torch.ones_like(metadata_enc) * 0.8).to(
                    metadata_enc.device
                )
                metadata_enc = metadata_enc * feature_mask

        if self.hidden is not None:
            # Detach current hidden
            self.hidden = self.hidden.detach()

            mask = torch.bernoulli(
                torch.ones_like(self.hidden) * (1 - N_HIDDEN_DROPOUT)
            ).to(self.hidden.device)
            self.hidden = self.hidden * mask

            self.hidden = self.context_mixer(
                torch.cat(
                    [
                        self.hidden,
                        metadata_enc.unsqueeze(0).repeat(self.hidden.shape[0], 1, 1),
                    ],
                    dim=2,
                )
            )
        else:
            self.hidden = metadata_enc.unsqueeze(0).repeat(
                self.decoder.num_layers, 1, 1
            )

        # Encode through MLP
        return metadata_enc

    def reset_hidden(self):
        if self.hidden is not None:
            self.hidden.detach()
            self.hidden = None

    def forward(
        self,
        categorical: torch.Tensor,
        numerical: torch.Tensor,
        target_payload: Optional[torch.Tensor] = None,
        max_length: int = N_MAX_PAYLOAD_LEN,
        teacher_forcing_ratio: float = N_TEACHER_FORCING_RATIO,
        hidden_teacher_forcing: bool = True,  # New parameter
    ):
        """
        Generate payload sequence based on metadata

        Args:
            categorical: Tensor of categorical features
            numerical: Tensor of numerical features
            target_payload: Optional target for teacher forcing (training)
            max_length: Maximum payload length to generate
            teacher_forcing_ratio: Probability of using teacher forcing
        """
        # Encode metadata context
        metadata_enc = self.encode_metadata(categorical, numerical)

        # Determine sequence length
        if target_payload is not None:
            seq_length = len(target_payload)
        else:
            # For inference, use the payload length feature or max_length
            seq_length = max_length

        # Initialize decoder input with SOS token
        decoder_input = (
            torch.ones(1, 1, self.embedding_size, device=categorical.device) * SOS
        )

        # Output storage
        outputs = torch.zeros(seq_length, VOCAB_DIM, device=categorical.device)
        predictions = torch.zeros(
            seq_length, dtype=torch.long, device=categorical.device
        )

        # Expanded metadata context for concatenation with each input byte
        metadata_expanded = metadata_enc.unsqueeze(0).expand(1, 1, -1)

        # Generate sequence
        for t in range(seq_length):
            # Combine byte embedding with metadata context
            decoder_input_combined = torch.cat(
                [decoder_input, metadata_expanded], dim=2
            )

            # Decode one step
            output, hidden_pred = self.decoder(decoder_input_combined, self.hidden)

            # Project to vocabulary distribution
            logits = self.output_projection(output.squeeze(0))
            outputs[t] = logits

            # Use own prediction
            if self.training:
                # During training, sample with temperature
                probs = F.softmax(logits / self.temperature, dim=1)
                predictions[t] = torch.multinomial(probs, 1).squeeze(1)
            else:
                # During inference, use argmax
                predictions[t] = logits.argmax(dim=1)

            # Sample next input
            if target_payload is not None and random.random() < teacher_forcing_ratio:
                # Teacher forcing - use actual target
                next_byte = target_payload[t]
            else:
                next_byte = torch.tensor(
                    predictions[t].item(), dtype=torch.long, device=DEVICE
                )

            if self.training and random.random() < N_MASK_PROB:
                next_byte = torch.tensor(MASK, dtype=torch.long, device=DEVICE)

            # NEW: Hidden state teacher forcing during training
            if self.training and hidden_teacher_forcing and target_payload is not None:
                # Always update hidden state with the correct target, regardless of what we use for next input
                correct_input = self.byte_embedding(
                    target_payload[t].unsqueeze(0)
                ).unsqueeze(1)
                correct_combined = torch.cat([correct_input, metadata_expanded], dim=2)

                # Run decoder again to get correct hidden state
                _, correct_hidden = self.decoder(correct_combined, self.hidden)

                # Use correct hidden state
                self.hidden = correct_hidden
            else:
                # Use the hidden state from prediction
                self.hidden = hidden_pred

            # Embed next byte for input
            decoder_input = self.byte_embedding(next_byte.unsqueeze(0)).unsqueeze(1)

        return outputs, predictions


### Helper functions ###
def train_model(csv_dir: str, num_epochs=N_NUM_EPOCHS):
    """Train the packet generator model"""
    train_loader, val_loader, test_loader = generate_loaders(
        csv_dir=csv_dir, epoch_num=0
    )

    # Initialize the model
    model = PacketGenerator(train_loader.cat_dims, numerical_dim=train_loader.num_dims)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=N_LR, weight_decay=N_WEIGHT_DECAY
    )
    criterion = LabelSmoothingCrossEntropy(
        smoothing=N_SMOOTHING
    )  # Label smoothing for robustness
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=N_MAX_LR,
        total_steps=num_epochs * len(train_loader),
        pct_start=0.1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Metrics #
    best_val_loss = float("inf")
    train_losses = list()
    val_losses = list()
    train_acces = list()
    val_acces = list()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        total_bytes = 0
        n_hidden_reliant = 0

        # Gradually decrease temperature and teacher forcing
        temperature = max(0.5, 1.0 - epoch / num_epochs)
        reset_probability = max(N_MEM_RESET_PROB, 0.9 - 0.8 * epoch / num_epochs)
        model.reset_hidden()
        # tf_ratio = max(N_TEACHER_FORCING_RATIO, 1.0 - epoch * 1.5 / num_epochs)
        tf_ratio = N_TEACHER_FORCING_RATIO
        model.temperature = temperature

        # Progressive truncation - start with very short sequences and gradually increase
        # This forces model to rely on hidden state rather than pattern memorization
        if epoch < num_epochs // 3:
            max_seq_len = min(8, N_MAX_PAYLOAD_LEN)  # Start with very short sequences
        elif epoch < 2 * num_epochs // 3:
            max_seq_len = min(16, N_MAX_PAYLOAD_LEN)  # Medium sequences
        else:
            max_seq_len = N_MAX_PAYLOAD_LEN  # Full sequences

        for i, packet in enumerate(train_loader):
            cat_features, num_features, payload = [b.to(device) for b in packet]

            if len(payload) == 0:
                continue

            if len(payload) > max_seq_len:
                payload = payload[:max_seq_len]

            # if random.random() < reset_probability:
            #     model.reset_hidden()

            optimizer.zero_grad()
            # Forward pass with scheduled teacher forcing
            logits, predictions = model(
                cat_features, num_features, payload, teacher_forcing_ratio=tf_ratio
            )

            # Compute loss and accuracy
            loss = criterion(logits.view(-1, VOCAB_DIM), payload.view(-1))
            hidden_reliance = hidden_reliance_loss(
                model, criterion, cat_features, num_features, payload, loss
            )
            if hidden_reliance == 0:
                n_hidden_reliant += 1

            hidden_loss = compute_hidden_state_regularization(model.hidden)

            tot_loss = loss + hidden_reliance + hidden_loss

            # Backpropagation
            tot_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Stats
            train_loss += tot_loss.item() * len(payload)
            correct = (predictions == payload).sum().item()
            train_acc += correct
            total_bytes += payload.numel()

            if DEBUG_MODE:
                print(f"Actual: {payload}\nPredictions: {predictions}")
                print_update(
                    mode="Train",
                    packet_num=i,
                    batch_loss=loss.item(),
                    batch_acc=correct / len(payload) if len(payload) > 0 else 0,
                    global_train_acc=train_acc / total_bytes,
                    global_train_loss=train_loss / total_bytes,
                    ratio_hidden_reliant=n_hidden_reliant / (i + 1),
                )

        if total_bytes > 0:
            train_loss /= total_bytes
            train_acc /= total_bytes
        else:
            train_loss = float("inf")
            train_acc = float("inf")

        train_losses.append(train_loss)
        train_acces.append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        total_val_bytes = 0
        n_hidden_reliant = 0

        with torch.no_grad():
            for i, packet in enumerate(val_loader):
                cat_features, num_features, payload = [b.to(device) for b in packet]

                if len(payload) == 0:
                    continue

                # Forward pass (no teacher forcing)
                logits, predictions = model(
                    cat_features, num_features, payload, teacher_forcing_ratio=0.0
                )

                # Compute loss and accuracy
                loss = criterion(logits.view(-1, VOCAB_DIM), payload.view(-1))
                hidden_reliance = hidden_reliance_loss(
                    model, criterion, cat_features, num_features, payload, loss
                )
                if hidden_reliance == 0:
                    n_hidden_reliant += 1
                hidden_loss = compute_hidden_state_regularization(model.hidden)

                tot_loss = loss + hidden_reliance + hidden_loss

                # Stats
                val_loss += tot_loss.item() * len(payload)
                correct = (predictions == payload).sum().item()
                val_acc += correct
                total_val_bytes += payload.numel()

                if DEBUG_MODE:
                    print(f"Actual: {payload}\nPredictions: {predictions}")
                    print_update(
                        mode="Validation",
                        batch_num=i,
                        batch_len=len(payload),
                        batch_loss=loss.item(),
                        batch_acc=correct / len(payload) if len(payload) > 0 else 0,
                        global_val_acc=val_acc / total_val_bytes,
                        global_val_loss=val_loss / total_val_bytes,
                        ratio_hidden_reliant=n_hidden_reliant / (i + 1),
                    )

        if total_val_bytes > 0:
            val_loss /= total_val_bytes
            val_acc /= total_val_bytes
        else:
            val_loss = float("inf")
            val_acc = float("inf")

        val_losses.append(val_loss)
        val_acces.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Temperature: {temperature:.2f}, TF Ratio: {tf_ratio:.2f}")

        # plot the losses and accuracies
        if DEBUG_MODE and epoch > num_epochs // 2:
            plot_metrics(
                train_losses,
                f"Training loss for {epoch}",
                x_label="Epochs",
                y_label="Epoch Loss",
            )
            plot_metrics(
                val_losses,
                f"Validation loss for {epoch}",
                x_label="Epochs",
                y_label="Epoch Loss",
            )
            plot_metrics(
                train_acces,
                f"Training acc for {epoch}",
                x_label="Epochs",
                y_label="Epoch acc",
            )
            plot_metrics(
                val_acces,
                f"Validation acc for {epoch}",
                x_label="Epochs",
                y_label="Epoch acc",
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        #     torch.save(
        #         {
        #             "epoch": epoch,
        #             "model_state_dict": model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "val_loss": val_loss,
        #         },
        #         f"packet_generator_best.pt",
        #     )

        # Regenerate the loaders for the next epoch
        train_loader, val_loader, test_loader = generate_loaders(
            csv_dir=csv_dir, epoch_num=epoch + 1
        )


def packet_it_generator(
    df_split: List[PacketDataset], epoch_num: int = N_NUM_EPOCHS - 1
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    @Description: Creates a batch stream of parsed packets from the given conversations

    @Notes:
        - We return if the end of a conversation is reached before the full batch lenght is reached

    @Returns: (cat_features_tensor, numerical_features_tensor, payloads_tensor)
    """
    # Use the epoch number to schedule how many packets will we will select from each conversation
    n_conv_packets = int(N_MAX_CONV_PACKETS * (1 + epoch_num) / N_NUM_EPOCHS)
    print(f"n_conv_packets: {n_conv_packets}")
    for df in df_split:

        for i, packet in enumerate(df):
            if i > n_conv_packets:
                break
            yield packet.cat_features, packet.numerical_features, torch.tensor(
                packet.payload
            )


def generate_loaders(csv_dir: str, epoch_num: int = N_NUM_EPOCHS - 1) -> Tuple[
    PacketIterator,
    PacketIterator,
    PacketIterator,
]:
    """
    @Description: Takes a general dataset, splits it into validation and training and then
    creates loaders for each data split

    @Notes:

    @Returns:
    """
    global conv_list, split_dict
    train_dfs = list()
    validation_dfs = list()
    test_dfs = list()

    train_len = 0
    validation_len = 0
    test_len = 0

    cat_dims = list()
    num_dims = 0
    # Load the dataset
    dfs = load_dfs_from_dir(csv_dir=csv_dir)
    for df in dfs:
        # Get the conversations splits
        splits = split_into_conversations(df, conv_list=conv_list)

        print(conv_list)
        conv_dfs = [
            PacketDataset(conv_df, n_convs=len(conv_list)) for conv_df in splits
        ]

        train, validation, test = split_convs(conv_dfs).values()

        cat_dims = train[0].cat_dims
        num_dims = train[0].num_dims

        train_dfs += train
        validation_dfs += validation
        test_dfs += test

        for t in train:
            train_len += len(t)

        for v in validation:
            validation_len += len(v)

        for ts in test:
            test_len += len(ts)

    assert isinstance(cat_dims, list), f"The cat dims must be a list of integers"
    assert isinstance(num_dims, int), f"The num dims must be an integer"

    # Now create batch generators for each
    return (
        PacketIterator(
            packet_it_generator(train_dfs, epoch_num),
            train_len,
            cat_dims=cat_dims,
            num_dims=num_dims,
        ),
        PacketIterator(
            packet_it_generator(validation_dfs, epoch_num),
            validation_len,
            cat_dims=cat_dims,
            num_dims=num_dims,
        ),
        PacketIterator(
            packet_it_generator(test_dfs, epoch_num),
            test_len,
            cat_dims=cat_dims,
            num_dims=num_dims,
        ),
    )


if __name__ == "__main__":

    csv_dir = "test_data"
    csv_dir = "/home/joel/Documents/laurier/URSA/research/datasets/mqtt-data/kaggle_mqtt_set/Data/PCAP/legit_cap_split/small_sample"
    # csv_dir = (
    #     "datasets/mqtt-data/kaggle_mqtt_set/Data/PCAP/legit_cap_split/small_sample"
    # )

    train_model(csv_dir=csv_dir)
