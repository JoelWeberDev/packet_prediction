"""
@Author: Joel Weber
@Date: 2025-07-18
@Description: Since the mqtt data is highly structured and repetetive within a conversation
most standard models quickly suffer from overfitting and learning a false gradient.
The repetetive template gru circumvents this by establishing embeddings for n templates for each
conversation and then matching that with context, meta data, and hidden states. The advantage here
is that we plan for repetition and use that to our advantage. It also forces generalization
at the architechural level because the model is trained to learn templates rather than go through
the payload predicting it byte by byte

@Notes:

@TODO:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Dict, Tuple, Optional, Iterator
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
    PacketIterator,
    diversity_loss,
    entropy_regularization,
    pattern_break_loss,
)


class AdaptivePacketGenerator(nn.Module):
    """
    Simpler model that adapts to patterns during inference
    without excessive complexity
    """

    def __init__(
        self,
        categorical_dims: List[int],
        numerical_dim: int,
        hidden_size: int = R_HIDDEN_SIZE,
        embedding_size: int = R_BYTE_EMBEDDED_SIZE,
        pattern_memory_size: int = R_PATTERN_SIZE,  # QQ what exactly does a hidden state vector represent?
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Metadata encoder
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(dim, min(embedding_size, (dim + 1) // 2))
                for dim in categorical_dims
            ]
        )

        # Count the number of embeddings that are required to embed the categorical variables
        cat_embed_dim = sum(
            min(embedding_size, (dim + 1) // 2) for dim in categorical_dims
        )
        metadata_dim = cat_embed_dim + numerical_dim

        # Enhanced metadata encoder with residual connections
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Byte embeddings with positional encoding
        self.byte_embedding = nn.Embedding(VOCAB_DIM, embedding_size)
        self.position_embedding = nn.Embedding(R_MAX_SEQ_LEN, embedding_size)

        # Core GRU decoder with multiple layers
        self.decoder = nn.GRU(
            input_size=embedding_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
        )

        # Anti-repetition mechanism - track recent outputs
        self.register_buffer("recent_outputs", torch.zeros(10, VOCAB_DIM))
        self.output_history_idx = 0

        # Diversity encouragement layers
        self.diversity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Output projection with mixture of experts
        self.num_experts = 4
        self.expert_gates = nn.Linear(hidden_size, self.num_experts)
        self.experts = nn.ModuleList(
            [nn.Linear(hidden_size, VOCAB_DIM) for _ in range(self.num_experts)]
        )

        # Enhanced pattern memory with diversification
        self.pattern_memory_size = pattern_memory_size
        self.register_buffer(
            "pattern_memory", torch.zeros(pattern_memory_size, hidden_size)
        )
        self.register_buffer("pattern_usage", torch.zeros(pattern_memory_size))
        self.register_buffer("pattern_confidence", torch.zeros(pattern_memory_size))
        self.register_buffer("pattern_diversity", torch.zeros(pattern_memory_size))

        # For inference adaptation
        self.adaptation_rate = R_ADAPTATION_RATE
        self.temperature = R_TEMPERATURE
        self.online_adaptation = False
        self.hidden = None
        self.position_counter = 0

    def encode_metadata(self, categorical, numerical):
        # Embed categorical features
        cat_embeds = [
            embed(categorical[i]) for i, embed in enumerate(self.cat_embeddings)
        ]
        cat_concat = torch.cat(cat_embeds, dim=0)

        # Combine with numerical features
        metadata_features = torch.cat([cat_concat, numerical], dim=0)

        # Encode through MLP
        return self.metadata_encoder(metadata_features)

    def reset_hidden(self):
        """Reset hidden state between conversations"""
        self.hidden = None

    def enable_adaptation(self, enable=True, rate=R_ADAPTATION_RATE):
        """Enable/disable online adaptation during inference"""
        self.online_adaptation = enable
        self.adaptation_rate = rate

    def reset_patterns(self):
        """Reset learned patterns - call at the start of new conversations"""
        self.pattern_memory.zero_()
        self.pattern_usage.zero_()
        self.pattern_confidence.zero_()
        self.pattern_diversity.zero_()

    def find_matching_pattern(self, query, threshold=0.7):
        """Find if current hidden state matches any stored pattern"""
        if self.pattern_usage.sum() == 0:
            return None, 0.0

        # Compute similarity with stored patterns
        similarities = F.cosine_similarity(
            query.unsqueeze(0), self.pattern_memory, dim=1
        )

        # Penalize overused patterns to encourage diversity
        usage_penalty = self.pattern_usage / (self.pattern_usage.sum() + 1e-8)
        diversity_bonus = self.pattern_diversity / (self.pattern_diversity.max() + 1e-8)

        # Adjust similarities based on usage and diversity
        adjusted_similarities = (
            similarities - 0.3 * usage_penalty + 0.1 * diversity_bonus
        )

        # Get best match
        best_idx = int(torch.argmax(adjusted_similarities).item())
        best_sim = similarities[
            best_idx
        ].item()  # Use original similarity for threshold

        if best_sim > threshold:
            return best_idx, best_sim
        return None, best_sim

    def update_pattern_memory(self, hidden_state, match_idx=None):
        """Update pattern memory with current hidden state and diversity tracking"""
        if not self.online_adaptation:
            return

        # If we found a match, update it
        if match_idx is not None:
            # Update existing pattern with momentum
            old_pattern = self.pattern_memory[match_idx].clone()
            self.pattern_memory[match_idx] = (
                1 - self.adaptation_rate
            ) * self.pattern_memory[match_idx] + self.adaptation_rate * hidden_state

            # Update diversity metric (how much the pattern is changing)
            pattern_change = F.cosine_similarity(
                old_pattern, self.pattern_memory[match_idx], dim=0
            )
            self.pattern_diversity[match_idx] = 0.9 * self.pattern_diversity[
                match_idx
            ] + 0.1 * (1 - pattern_change)

            self.pattern_usage[match_idx] += 1
            self.pattern_confidence[match_idx] = min(
                1.0, self.pattern_confidence[match_idx] + 0.1
            )
            return

        # No match found, store in least diverse or least used slot
        least_used = torch.argmin(
            self.pattern_usage + 0.5 * self.pattern_diversity
        ).item()
        self.pattern_memory[least_used] = hidden_state
        self.pattern_usage[least_used] = 1
        self.pattern_confidence[least_used] = 0.5
        self.pattern_diversity[least_used] = (
            1.0  # New patterns start with high diversity
        )

    def forward(self, categorical, numerical, target_payload=None):
        # Encode metadata
        metadata_enc = self.encode_metadata(categorical, numerical)

        # Initialize hidden state if needed
        if self.hidden is None:
            self.hidden = torch.zeros(
                2, 1, self.hidden_size, device=metadata_enc.device  # num_layers=2
            )

        # Determine sequence length
        if target_payload is not None:
            seq_length = len(target_payload)
        else:
            seq_length = R_MAX_SEQ_LEN  # Default max length

        # Check if current state matches any known pattern
        match_idx, match_similarity = self.find_matching_pattern(
            self.hidden[-1].squeeze()
        )

        # If strong pattern match found during inference, add diversity noise
        if match_idx is not None and match_similarity > R_SIMILARITY_THRESH:
            # Instead of blending, add controlled noise to break patterns
            noise_scale = 0.1 * (
                1 - match_similarity
            )  # More noise for stronger matches
            diversity_noise = torch.randn_like(self.hidden) * noise_scale
            self.hidden = self.hidden + diversity_noise

        # Initialize output containers
        outputs = []
        predictions = torch.zeros(
            seq_length, dtype=torch.long, device=categorical.device
        )

        # Start with SOS token
        input_byte = torch.ones(1, 1, dtype=torch.long, device=categorical.device) * SOS
        prior_hidden = self.hidden.detach().clone()

        # Generate sequence with anti-repetition mechanism
        for t in range(seq_length):
            # Add positional encoding
            pos_emb = self.position_embedding(
                torch.tensor([t % R_MAX_SEQ_LEN], device=categorical.device)
            )

            # Embed input byte
            byte_emb = self.byte_embedding(input_byte).view(1, 1, -1)
            byte_emb = byte_emb + pos_emb.unsqueeze(0)

            # Combine with metadata
            decoder_input = torch.cat(
                [byte_emb, metadata_enc.unsqueeze(0).unsqueeze(0)], dim=2
            )

            # Pass through GRU
            output, self.hidden = self.decoder(decoder_input, self.hidden)

            # Mixture of experts output
            gating_weights = F.softmax(
                self.expert_gates(output.squeeze(0)), dim=-1
            ).reshape(self.num_experts, -1)
            expert_outputs = torch.stack(
                [expert(output.squeeze(0)) for expert in self.experts]
            ).squeeze(1)
            logits = torch.sum(gating_weights * expert_outputs, dim=0).unsqueeze(0)

            outputs.append(logits)

            # Make prediction with temperature and diversity sampling
            if self.training:
                # Use nucleus sampling for more diverse outputs
                probs = F.softmax(logits / self.temperature, dim=-1)

                # Nucleus sampling (top-p)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                nucleus_mask = cumulative_probs <= R_NUCLEUS_P
                nucleus_mask[..., 0] = True  # Always keep the top token

                filtered_probs = torch.zeros_like(probs)
                filtered_probs.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
                filtered_probs = filtered_probs / filtered_probs.sum(
                    dim=-1, keepdim=True
                )

                pred = torch.multinomial(filtered_probs, 1).squeeze()
            else:
                pred = logits.argmax(dim=1)

            predictions[t] = pred

            # Next input (teacher forcing if training)
            if self.training and target_payload is not None:
                input_byte = target_payload[t : t + 1].unsqueeze(0)
            else:
                input_byte = pred.unsqueeze(0).unsqueeze(0)

        # Update hidden state with target sequence for better learning
        if target_payload is not None:
            self.hidden = prior_hidden
            for t_byte in target_payload:
                # Add positional encoding
                pos_idx = torch.tensor(
                    [self.position_counter % R_MAX_SEQ_LEN], device=categorical.device
                )
                pos_emb = self.position_embedding(pos_idx)

                # Embed input byte
                byte_emb = self.byte_embedding(t_byte.unsqueeze(0)).view(1, 1, -1)
                byte_emb = byte_emb + pos_emb.unsqueeze(0)

                # Combine with metadata
                decoder_input = torch.cat(
                    [byte_emb, metadata_enc.unsqueeze(0).unsqueeze(0)], dim=2
                )

                # Pass through GRU
                output, self.hidden = self.decoder(decoder_input, self.hidden)
                self.position_counter += 1

        # After sequence generation during inference, update pattern memory
        if self.online_adaptation:
            # Update existing pattern or create new one
            self.update_pattern_memory(
                self.hidden[-1].detach().clone().squeeze(0), match_idx
            )

        return torch.cat(outputs, dim=0), predictions


### Training helper functions ###
def train_model(csv_dir: str, num_epochs=N_NUM_EPOCHS):
    """Train the packet generator model"""
    train_loader, val_loader, test_loader = generate_loaders(
        csv_dir=csv_dir, epoch_num=0
    )

    # Initialize the model
    model = AdaptivePacketGenerator(
        train_loader.cat_dims, numerical_dim=train_loader.num_dims
    )

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

    model.enable_adaptation()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_hidden_rel_loss = 0.0
        total_bytes = 0
        n_hidden_reliant = 0

        # Gradually decrease temperature and teacher forcing
        # Weight all the loss functions based on the current epoch
        # losses are in order [logits loss, hidden internal change loss, hidden reliance loss]
        epoch_ratio = epoch / (num_epochs - 1)
        loss_weights = torch.tensor(
            [0.8 * epoch_ratio, 0.2, 0.8 * (1 - epoch_ratio)],
            dtype=torch.float32,
            device=DEVICE,
        )

        temperature = max(0.5, 1.0 - epoch_ratio)
        model.temperature = temperature
        h_loss_margin = min(0.5, 0.1 + 0.4 * epoch_ratio)
        model.temperature = temperature
        # reset_probability = max(N_MEM_RESET_PROB, 0.9 - 0.8 * epoch_ratio)
        # tf_ratio = max(N_TEACHER_FORCING_RATIO, 1.0 - epoch * 1.5 / num_epochs)
        model.reset_hidden()  # Default the hidden back to an original state
        model.reset_patterns()  # Create fresh patterns for each conversation
        tf_ratio = N_TEACHER_FORCING_RATIO

        # Progressive curriculum - start with very short sequences and gradually increase
        # This forces model to rely on hidden state rather than pattern memorization
        if epoch < num_epochs // 4:
            max_seq_len = min(8, N_MAX_PAYLOAD_LEN)  # Start with very short sequences
            temp_multiplier = 1.5  # Higher temperature for more exploration
        elif epoch < num_epochs // 2:
            max_seq_len = min(16, N_MAX_PAYLOAD_LEN)  # Medium sequences
            temp_multiplier = 1.2
        elif epoch < 3 * num_epochs // 4:
            max_seq_len = min(32, N_MAX_PAYLOAD_LEN)  # Longer sequences
            temp_multiplier = 1.0
        else:
            max_seq_len = N_MAX_PAYLOAD_LEN  # Full sequences
            temp_multiplier = 0.8  # Lower temperature for more focused generation

        model.temperature = temperature * temp_multiplier

        # Track conversation changes to reset memory appropriately
        last_conversation = None
        packets_in_conversation = 0

        for i, packet in enumerate(train_loader):
            cat_features, num_features, payload = [b.to(device) for b in packet]

            if len(payload) == 0:
                continue

            # Check if we're in a new conversation (implement based on your data structure)
            # This is a placeholder - you'll need to adapt based on how conversations are marked
            current_conversation = getattr(packet, "conversation_id", None)
            if current_conversation != last_conversation:
                model.reset_hidden()
                model.reset_patterns()
                packets_in_conversation = 0
                last_conversation = current_conversation

            packets_in_conversation += 1

            # Reset occasionally within long conversations to prevent overfitting
            if packets_in_conversation > 50 and random.random() < 0.1:
                model.reset_hidden()

            if len(payload) > max_seq_len:
                payload = payload[:max_seq_len]

            # Data augmentation: Random payload truncation and noise injection
            if random.random() < R_AUGMENT_PROB:
                if random.random() < R_TRUNCATE_PROB:  # Random truncation
                    trunc_len = random.randint(
                        max(1, int(len(payload) * R_MIN_TRUNCATE_RATIO)), len(payload)
                    )
                    payload = payload[:trunc_len]
                else:  # Add noise to a few bytes
                    payload = payload.clone()
                    n_noise = random.randint(1, min(R_MAX_NOISE_BYTES, len(payload)))
                    noise_indices = random.sample(range(len(payload)), n_noise)
                    for idx in noise_indices:
                        payload[idx] = random.randint(0, 255)

            # Reset hidden state randomly to break dependencies
            if random.random() < R_HIDDEN_RESET_PROB:
                model.reset_hidden()

            # Store hidden states for pattern analysis
            hidden_states = []

            optimizer.zero_grad()
            # Forward pass with scheduled teacher forcing
            logits, predictions = model.forward(cat_features, num_features, payload)

            # Collect hidden states during forward pass for pattern analysis
            if hasattr(model, "hidden") and model.hidden is not None:
                hidden_states.append(model.hidden[-1].detach().clone())

            # Compute primary loss
            loss = criterion(logits.view(-1, VOCAB_DIM), payload.view(-1))

            # Hidden reliance loss
            hidden_reliance = hidden_reliance_loss(
                model,
                criterion,
                cat_features,
                num_features,
                payload,
                loss,
                scale=R_HIDDEN_RELIANCE_LOSS_SCALE,
                margin=h_loss_margin,
            )

            # Hidden state regularization
            hidden_loss = compute_hidden_state_regularization(model.hidden)

            # New diversity and pattern breaking losses
            diversity_loss_val = diversity_loss(predictions, window_size=5)
            entropy_loss = entropy_regularization(logits, target_entropy=3.0)
            pattern_loss = (
                pattern_break_loss(hidden_states)
                if len(hidden_states) > 1
                else torch.tensor(0.0, device=device)
            )

            # Progressive loss weighting
            if epoch < num_epochs // 4:
                # Early training: Focus on diversity and pattern breaking
                loss_vec = torch.stack(
                    [
                        loss * 0.3,
                        hidden_loss * 0.1,
                        hidden_reliance * 0.1,
                        diversity_loss_val * 0.3,
                        entropy_loss * 0.2,
                        pattern_loss * 0.1,
                    ]
                )
            elif epoch < num_epochs // 2:
                # Mid training: Balance all losses
                loss_vec = torch.stack(
                    [
                        loss * 0.5,
                        hidden_loss * 0.15,
                        hidden_reliance * 0.15,
                        diversity_loss_val * 0.1,
                        entropy_loss * 0.05,
                        pattern_loss * 0.05,
                    ]
                )
            else:
                # Late training: Focus on accuracy with some diversity
                loss_vec = torch.stack(
                    [
                        loss * 0.7,
                        hidden_loss * 0.1,
                        hidden_reliance * 0.1,
                        diversity_loss_val * 0.05,
                        entropy_loss * 0.03,
                        pattern_loss * 0.02,
                    ]
                )

            tot_loss = loss_vec.sum()

            # Backpropagation with gradient clipping
            tot_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=0.5
            )  # Tighter clipping
            optimizer.step()
            scheduler.step()

            # Stats
            if hidden_reliance.item() < 0.01:
                n_hidden_reliant += 1
            train_hidden_rel_loss += hidden_reliance.item()
            train_loss += tot_loss.item() * len(payload)
            correct = (predictions == payload).sum().item()
            train_acc += correct
            total_bytes += payload.numel()

            if DEBUG_MODE:
                print(f"Actual: {payload}\nPredictions: {predictions}")
                print_update(
                    mode="Train",
                    epoch=epoch,
                    packet_num=i,
                    packet_loss=tot_loss.item(),
                    loss_vect=loss_vec,
                    packet_acc=correct / len(payload) if len(payload) > 0 else 0,
                    global_train_acc=train_acc / total_bytes,
                    global_train_loss=train_loss / total_bytes,
                    global_train_hidden_loss=train_hidden_rel_loss / total_bytes,
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
        val_hidden_rel_loss = 0.0
        total_val_bytes = 0
        n_hidden_reliant = 0

        with torch.no_grad():
            for i, packet in enumerate(val_loader):
                cat_features, num_features, payload = [b.to(device) for b in packet]

                if len(payload) == 0:
                    continue

                # Forward pass (no teacher forcing)
                logits, predictions = model.forward(cat_features, num_features, payload)

                # Compute loss and accuracy
                loss = criterion(logits.view(-1, VOCAB_DIM), payload.view(-1))
                hidden_reliance = hidden_reliance_loss(
                    model,
                    criterion,
                    cat_features,
                    num_features,
                    payload,
                    loss,
                    scale=R_HIDDEN_RELIANCE_LOSS_SCALE,
                    margin=h_loss_margin,
                )
                val_hidden_rel_loss += hidden_reliance
                if hidden_reliance == 0:
                    n_hidden_reliant += 1
                hidden_loss = compute_hidden_state_regularization(model.hidden)

                # if epoch < num_epochs // 4:
                #     loss_vec = torch.tensor([0, 0, hidden_reliance], dtype=torch.float32)
                # else:
                #     loss_vec = torch.tensor([loss, hidden_loss, hidden_reliance], dtype=torch.float32)

                loss_vec = torch.stack([loss, hidden_loss, hidden_reliance])

                # Assign progressive weighting to the losses
                tot_loss = loss_vec.dot(loss_weights)

                # Stats
                val_loss += tot_loss.item() * len(payload)
                correct = (predictions == payload).sum().item()
                val_acc += correct
                total_val_bytes += payload.numel()

                if DEBUG_MODE:
                    print(f"Actual: {payload}\nPredictions: {predictions}")
                    print_update(
                        mode="Validation",
                        epoch=epoch,
                        batch_num=i,
                        payload_len=len(payload),
                        payload_loss=tot_loss.item(),
                        loss_vect=loss_vec,
                        payload_acc=correct / len(payload) if len(payload) > 0 else 0,
                        global_val_acc=val_acc / total_val_bytes,
                        global_val_loss=val_loss / total_val_bytes,
                        global_val_hidden_loss=val_hidden_rel_loss / total_val_bytes,
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
