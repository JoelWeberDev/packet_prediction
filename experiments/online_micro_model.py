"""
@Author: Joel Weber
@Date: 2025-07-25
@Description: This is a completely different concept that chooses to address the overfitting issue
by embracing it with online learning. The architechure is setup to have support components that will
be frozen at runtime and then an intenal model such as a GRU, LSTM, or ESN actively being updated
as the conversation progresses.

@Notes:
- The fundamental issue will be, how do we train the support components to rely on the internal
micro model rather than themselves memorizing the training data.
    - My proposed solution is to train the support architechure at the conversation level and
    the micro model at the packet level.
- Loss function design:
    - Training:
        - The train true to its name is for training the components for the model that will
        remain fixed throughout the inference process.
        - Training will happen at the conversation level. Each time we pass through a training
        conversation we will assess the model's ability to learn throughout the progression
        of the conversation
        - Conversation sizes will start very small with large and abrupt adjustments, and it
        will increase in lenght and granulariy as preformance improves.
        - The progressive loss function defined in the helper function looks at where the model
        began with its prediciton loss how it progressed and uses that to establish a gradient
- Dealing with the 2 step training
    1. Use a differentiable optimizer that we can actually run BP through
    2. Run all the support training after the inference stage
        - This does not work because by nature the support architechure must learn to accomodate
        a moving target, the micro gru rather than itself overfit to the particular conversation.
        Training the model when the micro gru is fixed will simply result in memorization of the
        conversation in the support parameters.

@Questions
    - Should the byte embedded be included in the inference time training?

@TODO:
    - Define a global random seed so that synchronous models will not be thrown off by random
    noisy initial conditions.
    - Add a scheduler for both the support component training and the micro gru
@Next Steps 2025-06-29
    - Periodically sample loss values for the the support training while in training mode
    - Get to the point that the model is actually beginning to converge.
    - Improve the efficiency of the model. At the moment it is very large and very slow
    - Run in just inference mode to demonstrate the model's ability
"""

### Python imports ###
import torch
import torch.nn as nn
import higher
from typing import List
from dataclasses import dataclass, field
from copy import deepcopy

### Local imports ###
from modules.CONSTANTS import *
from modules.custom_datasets import PacketDataset, ParsedPacket
from modules.helper_functions import (
    print_update,
    plot_metrics,
    conversation_tradjectory_loss,
    conversation_trajectory_loss_simple,
    get_memory,
    LabelSmoothingCrossEntropy,
    PacketItGenerator,
)


### Helper structures ###
@dataclass
class PacketPrediction:
    logits: torch.Tensor
    preds: torch.Tensor


@dataclass
class ConvResults:
    packet_losses: List[torch.Tensor]
    packet_accs: List[int]


@dataclass
class EpochResults:
    tot_loss: float = 0.0
    tot_acc: float = 0.0
    tot_conv_loss: float = 0
    n_packets: int = 0
    n_convs: int = 0

    ### Getters ###
    @property
    def avg_loss(self):
        return self.tot_loss / self.n_packets if self.n_packets > 0 else float("inf")

    @property
    def avg_acc(self):
        return self.tot_acc / self.n_packets if self.n_packets > 0 else float("inf")

    @property
    def avg_conv_loss(self):
        return self.tot_conv_loss / self.n_convs if self.n_convs > 0 else float("inf")

    ### Generics ###
    def __len__(self):
        return self.n_packets

    def __str__(self):
        return f"""
            avg loss: {self.avg_loss}
            avg acc: {self.avg_acc}
            avg conv loss: {self.avg_conv_loss}
            num packets: {self.n_packets}
            num convs: {self.n_convs}
        """


@dataclass
class ModelMetrics:
    loss: List[float] = field(default_factory=list)
    acc: List[float] = field(default_factory=list)

    @property
    def avg_loss(self):
        return np.mean(self.avg_loss) if len(self.loss) > 0 else float("inf")

    @property
    def avg_acc(self):
        return np.mean(self.avg_acc) if len(self.acc) > 0 else float("inf")


class MetadataEmbedder(nn.Module):
    """
    @Description: Embeds categorical and numerical features into a fixed size representation

    @Notes:

    """

    def __init__(
        self,
        categorical_dims: List[int],
        numerical_dim: int,
    ):
        super().__init__()

        # 1. Metadata Encoding
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(dim, min(O_CAT_MAX_EMB_DIM, (dim + 1) // 2))
                for dim in categorical_dims
            ]
        )

        cat_embed_dim = sum(
            min(O_CAT_MAX_EMB_DIM, (dim + 1) // 2) for dim in categorical_dims
        )
        metadata_dim = cat_embed_dim + numerical_dim

        # Metadata MLP with normalization
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, O_MD_HIDDEN_DIM),
            nn.LayerNorm(O_MD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(O_MD_DROPOUT),
            nn.Linear(O_MD_HIDDEN_DIM, O_MD_HIDDEN_DIM),
            nn.LayerNorm(O_MD_HIDDEN_DIM),
            nn.ReLU(),
        )  # Inference Frozen

    def get_embeddings(self, cat_f: torch.Tensor, num_f: torch.Tensor) -> torch.Tensor:
        cat_f = cat_f.to(device=DEVICE)
        num_f = num_f.to(device=DEVICE)
        # Embed categorical features
        cat_embeds = [embed(cat_f[i]) for i, embed in enumerate(self.cat_embeddings)]
        cat_concat = torch.cat(cat_embeds, dim=0)

        # Combine with numerical features
        metadata_features = torch.cat([cat_concat, num_f], dim=0)

        # Encode through MLP
        return self.metadata_encoder(metadata_features)


# Create a packet embedded here for context inputting
class PacketEmbedder(nn.Module):
    """
    @Description: Takes the meta data along with a payload and embeds it into a fixed size representation for the purpose
    to provide a context that is not reliant on the hidden state memory for the next packet prediciton.

    @Notes:
        - Only works on single inputs.
        - Uses a gru to embed any length of byte sequence

    """

    def __init__(
        self, metadata_embedder: MetadataEmbedder, byte_embedder: nn.Embedding
    ):
        super().__init__()

        self.metadata_embedder = metadata_embedder
        self.byte_embedder = byte_embedder

        # Payload embedder
        self.payload_embedder = nn.GRU(
            input_size=O_BYTE_EMB_DIM,
            hidden_size=O_PACKET_REP_DIM // 2,  # for bidirectionality
            num_layers=O_PACKET_EMB_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=O_PACKET_EMB_DROPOUT,
        )

        # Metadata mixer
        self.packet_combiner = nn.Sequential(
            nn.Linear(O_PACKET_REP_DIM + O_MD_HIDDEN_DIM, O_PACKET_REP_DIM),
            nn.ReLU(),
            nn.Dropout(O_PACKET_EMB_DROPOUT),
        )

    def get_embeddings(
        self, cat_f: torch.Tensor, num_f: torch.Tensor, payload: torch.Tensor
    ) -> torch.Tensor:
        metadata_emb = self.metadata_embedder.get_embeddings(
            cat_f.to(device=DEVICE), num_f.to(device=DEVICE)
        )

        byte_embs = self.byte_embedder(payload)

        output, _ = self.payload_embedder(byte_embs)

        last_state = output[-1, :]  # gets the state of the last step

        assert last_state.shape == (
            O_PACKET_REP_DIM,
        ), f"The packet shape of {last_state.shape} must equal {(O_PACKET_REP_DIM,)}"

        return self.packet_combiner(
            torch.cat([last_state, metadata_emb], dim=-1).to(device=DEVICE)
        )


class OnlinePacketPredictor(nn.Module):
    """
    @Description: Model implementing online learning at the conversation level for predicting
    the next packet.

    @Notes:

    @TODO:
        - Should I add compress the input size? It is rather large. Perhaps I should mix the context
        and metadata with the iternal hidden state.
        - Create a packet vocabulary similar to the byte embeddings that takes the embedded packet
        and then runs it through a fixed size embedder to classify that packet
        - Test with temperature selection using a fixed seed

    """

    inference_component_names = ("micro_byte_gru", "output_projection")
    support_component_names = ("metadata_embedder", "packet_embedder", "byte_embedder")

    def __init__(self, metadata_embedder: MetadataEmbedder):
        super().__init__()

        self.metadata_embedder = metadata_embedder

        # Byte Embeddings
        self.byte_embedder = nn.Embedding(VOCAB_DIM, O_BYTE_EMB_DIM)  # Inference Frozen

        # Create the packet embedder
        self.packet_embedder = PacketEmbedder(
            metadata_embedder=metadata_embedder, byte_embedder=self.byte_embedder
        )

        # Input size for each step
        self.input_size = (
            (O_PACKET_REP_DIM * O_PACKET_CTX_LEN)
            + (O_BYTE_CTX_LEN * O_BYTE_EMB_DIM)
            + O_MD_HIDDEN_DIM
        )

        # Micro model GRU
        self.micro_byte_gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=O_HIDDEN_SIZE,
            num_layers=O_NUM_LAYERS,
            dropout=O_DROPOUT,
            batch_first=True,
        )

        # Output Projection
        self.output_projection = nn.Linear(O_HIDDEN_SIZE, VOCAB_DIM)

        self.hidden = None
        self.temperature = O_DEFAULT_TEMP

    def forward(
        self, packet_ctx: List[ParsedPacket], target_packet: ParsedPacket
    ) -> PacketPrediction:
        """
        @Description: Auto regressively produces a packet prediction based on the context

        @Notes:
            - As an initial experiment the context data will simply be repeated, but if this
            model proves to be effective embedding the context at the packet level is an opt
            that should be done.
            -

        @Returns:
        """
        cat_f = target_packet.cat_features
        num_f = target_packet.numerical_features
        target_payload = target_packet.payload

        # Embed the packet context
        packet_ctx_emb = torch.cat(
            [
                self.packet_embedder.get_embeddings(
                    packet.cat_features.to(device=DEVICE),
                    packet.numerical_features.to(device=DEVICE),
                    torch.tensor(packet.payload, dtype=torch.long, device=DEVICE),
                )
                for packet in packet_ctx
            ],
            dim=-1,
        )

        # Embed the metadata
        metadata_emb = self.metadata_embedder.get_embeddings(cat_f=cat_f, num_f=num_f)

        # Create a byte context padding tensor
        ctx_payload = self.byte_embedder(
            torch.tensor(
                [MASK] * (O_BYTE_CTX_LEN - 1) + [SOS] + target_payload,
                dtype=torch.long,
                device=DEVICE,
            )
        )

        payload_len = len(target_payload)

        # Prepare empty tensors to store the results
        res_logits = torch.empty(
            (payload_len, VOCAB_DIM), dtype=torch.float32, device=DEVICE
        )
        preds = torch.empty(payload_len, dtype=torch.long, device=DEVICE)

        for i in range(payload_len):
            byte_ctx = ctx_payload[i : i + O_BYTE_CTX_LEN].flatten()

            # Generate the lstm input
            lstm_input = torch.cat(
                [packet_ctx_emb, metadata_emb, byte_ctx], dim=-1
            ).unsqueeze(0)

            output, self.hidden = self.micro_byte_gru(
                lstm_input, self.hidden.detach() if self.hidden is not None else None
            )

            # Create a projection
            logits = self.output_projection(output)

            res_logits[i, :] = logits

            preds[i] = logits.argmax(dim=-1)

        return PacketPrediction(logits=res_logits, preds=preds)

    ### Helper functions ###
    def get_component_groups(self):
        """
        @Description: Returns the model parameter group splits for both support training and
        micro gru model training

        @Notes:

        @Returns:
        """
        inference_params = list()
        support_params = list()

        for name, param in self.named_parameters():
            if any(component in name for component in self.inference_component_names):
                inference_params.append(param)
            else:
                support_params.append(param)

        return inference_params, support_params

    def train_mode_support_only(self):
        """Set only support components to training mode"""
        self.train()
        for name, param in self.named_parameters():
            if any(component in name for component in self.support_component_names):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def train_mode_micro_only(self):
        """Set only micro components to training mode"""
        self.train()
        for name, param in self.named_parameters():
            if any(component in name for component in self.inference_component_names):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def conv_reset(self):
        self.hidden = None
        self.micro_byte_gru.reset_parameters()
        self.output_projection.reset_parameters()


### Training and analysis functions ###
def run_conv(
    model: OnlinePacketPredictor,
    optimizer,
    criterion,
    conv_df: PacketDataset,
    max_conv_len: float,
    train: bool = False,
) -> ConvResults:
    # prep to train again

    packet_ctx = []
    packet_losses = []
    packet_accs = []

    with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
        assert isinstance(
            fmodel, OnlinePacketPredictor
        ), f"functional model has type {type(fmodel)}, but expected {type(model)}"

        for packet in conv_df:
            if len(packet.payload) == 0:
                continue

            if len(packet_ctx) == O_PACKET_CTX_LEN:
                target_payload = torch.tensor(
                    packet.payload, dtype=torch.long, device=DEVICE
                )

                results = fmodel.forward(packet_ctx, packet)
                loss = criterion(
                    results.logits.view(-1, VOCAB_DIM), target_payload.view(-1)
                )

                if train:
                    # torch.nn.utils.clip_grad_norm_(fmodel.parameters(), max_norm=1.0)
                    diffopt.step(loss)

                # Store results
                packet_losses.append(loss)
                correct = (results.preds == target_payload).sum().item()
                packet_accs.append(correct / len(packet.payload))

                if len(packet_losses) >= max_conv_len:
                    break

            packet_ctx.append(packet)
            packet_ctx = packet_ctx[-O_PACKET_CTX_LEN:]

    return ConvResults(packet_losses, packet_accs)


def train_model(csv_dir: str, num_epochs=N_NUM_EPOCHS):
    """Train the packet generator model"""
    packet_generator = PacketItGenerator(csv_dir)
    data_split_dict = packet_generator.generate_conv_loaders(csv_dir)

    # Initialize the model
    metadata_embeder = MetadataEmbedder(
        data_split_dict["train"][0].cat_dims, data_split_dict["train"][0].num_dims
    )
    model = OnlinePacketPredictor(metadata_embedder=metadata_embeder)

    # Create optimizer for inference time parameters
    inference_params, support_params = model.get_component_groups()
    inference_optim = torch.optim.Adam(
        inference_params, lr=O_INFERENCE_LR, weight_decay=O_WEIGHT_DECAY
    )
    support_optim = torch.optim.AdamW(
        support_params, lr=O_LR, weight_decay=O_WEIGHT_DECAY
    )

    criterion = LabelSmoothingCrossEntropy(
        smoothing=O_SMOOTHING
    )  # Label smoothing for robustness
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        support_optim,
        max_lr=O_MAX_LR,
        total_steps=num_epochs * len(data_split_dict["train"]),
        pct_start=0.1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Metrics #
    model_metrics = dict()
    for mode in data_split_dict.keys():
        model_metrics[mode] = ModelMetrics()

    for epoch in range(num_epochs):
        # Training phase
        packet_generator.n_conv_packets = (
            N_MAX_CONV_PACKETS  # Permit the max number of conv packets
        )

        # Gradually decrease temperature and teacher forcing
        temperature = max(0.5, 1.0 - epoch / num_epochs)
        max_conv_len = (
            O_MAX_CONV_LEN * (epoch + 1) / num_epochs
        )  # TBD! should I add NL scheduling?
        model.temperature = temperature

        epoch_results = dict()

        for mode, conv_dfs in data_split_dict.items():
            mode_results = EpochResults()
            epoch_results[mode] = mode_results

            for conv_df in conv_dfs:
                # Assess the model's preformance
                # Reset the model for the new conversation
                model.train_mode_micro_only()
                model.conv_reset()

                conv_results = run_conv(
                    model,
                    inference_optim,
                    criterion,
                    conv_df,
                    max_conv_len,
                    train=(mode == "train"),
                )
                packet_losses = conv_results.packet_losses
                packet_accs = conv_results.packet_accs

                if len(packet_losses) > 0:
                    # TBD!! what is a good loss function for this issue?
                    conv_loss = conversation_trajectory_loss_simple(
                        torch.stack(packet_losses)
                    )

                    # Loss and feedback stuff
                    if mode == "train":
                        model.train_mode_support_only()
                        support_optim.zero_grad()
                        conv_loss.backward()
                        support_optim.step()
                        scheduler.step()

                    mode_results.tot_conv_loss += conv_loss.item()
                    mode_results.n_convs += 1
                    mode_results.n_packets += len(packet_losses)

                    loss_mean = float(np.mean([loss.item() for loss in packet_losses]))
                    acc_mean = float(np.mean(packet_accs))
                    mode_results.tot_loss += loss_mean
                    mode_results.tot_acc += acc_mean

                    # Print results
                    if DEBUG_MODE:
                        print_update(
                            mode=mode,
                            epoch_number=epoch,
                            packet_loss_mean=loss_mean,
                            packet_acc_mean=acc_mean,
                            conv_packet_loss=conv_loss.item(),
                            epoch_mode_results=str(mode_results),
                        )

            epoch_results[mode] = mode_results

            model_metrics[mode].loss.append(mode_results.avg_loss)
            model_metrics[mode].acc.append(mode_results.avg_acc)

        # plot the losses and accuracies
        if DEBUG_MODE and epoch > num_epochs // 2:
            for key, metrics in model_metrics.items():
                plot_metrics(
                    metrics.avg_loss,
                    f"{key} loss for {epoch}",
                    x_label="Epochs",
                    y_label="Epoch Loss",
                )
                plot_metrics(
                    metrics.avg_acc,
                    f"{key} acc for {epoch}",
                    x_label="Epochs",
                    y_label="Epoch acc",
                )

        # Regenerate the loaders for the next epoch
        data_split_dict = packet_generator.generate_conv_loaders(csv_dir)


if __name__ == "__main__":

    csv_dir = "test_data"
    # csv_dir = (
    #     "../datasets/mqtt-data/kaggle_mqtt_set/Data/PCAP/legit_cap_split/small_sample"
    # )
    # csv_dir = (
    #     "datasets/mqtt-data/kaggle_mqtt_set/Data/PCAP/legit_cap_split/small_sample"
    # )

    train_model(csv_dir=csv_dir)
