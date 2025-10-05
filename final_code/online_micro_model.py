"""
@Author: Joel Weber
@Date: 2025-07-25
@Description: The online learning model is designed to learn both at the conversation
level and the protocol level. During inference the model is always learning at the
packet level. Then once the conversation is done and the model is in training
we train it based on its overall preformance and progression throughout the
conversation. This forces the inference time frozen components to learn abstractions
of the protocol rather than simply trying to memorize every conversation.
When a new conversation is started, the internal GRU and ouput projection are
reset so they can learn the new conversation from a fresh start
"""

### Python imports ###
import os
import time
import torch
import torch.nn as nn
import higher
from typing import List

### Local imports ###
from CONSTANTS import *
from custom_datasets import PacketDataset, ParsedPacket
from helper_functions import (
    print_update,
    pkl_write_model,
    conversation_trajectory_loss_simple,
    conversation_tradjectory_loss,
    process_model_metrics,
    get_num_model_params,
    get_used_memory,
    LabelSmoothingCrossEntropy,
    FocusLoss,
    PacketItGenerator,
    PacketPrediction,
    ConvResults,
    EpochResults,
    ModelMetrics,
)


### Helper structures ###


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

        print(
            f"Metadata Embedder num params: {get_num_model_params(self.metadata_embedder)}"
        )

        # Byte Embeddings
        self.byte_embedder = nn.Embedding(VOCAB_DIM, O_BYTE_EMB_DIM)  # Inference Frozen

        print(f"Byte Embedder num params: {get_num_model_params(self.byte_embedder)}")

        # Create the packet embedder
        self.packet_embedder = PacketEmbedder(
            metadata_embedder=metadata_embedder, byte_embedder=self.byte_embedder
        )

        print(
            f"Packet Embedder num params: {get_num_model_params(self.packet_embedder)}"
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

        # Micro model LSTM
        # self.micro_byte_gru = nn.LSTM(
        #     input_size=self.input_size,
        #     hidden_size=O_HIDDEN_SIZE,
        #     num_layers=O_NUM_LAYERS,
        #     dropout=O_DROPOUT,
        #     batch_first=True,
        # )

        # Micro model RNN
        self.micro_byte_gru = nn.RNN(
            input_size=self.input_size,
            hidden_size=O_HIDDEN_SIZE,
            num_layers=O_NUM_LAYERS,
            dropout=O_DROPOUT,
            batch_first=True,
        )

        print(f"Micro GRU size: {get_num_model_params(self.micro_byte_gru)}")

        # Micro model ESN
        # TODO implement ESN

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

            # Generate the input
            model_input = torch.cat(
                [packet_ctx_emb, metadata_emb, byte_ctx], dim=-1
            ).unsqueeze(0)

            # GRU version
            output, self.hidden = self.micro_byte_gru(
                model_input, self.hidden.detach() if self.hidden is not None else None
            )

            # LSTM version
            # output, self.hidden = self.micro_byte_gru(
            #     model_input,
            #     (
            #         (self.hidden[0].detach(), self.hidden[1].detach())
            #         if self.hidden is not None
            #         else None
            #     ),
            # )

            # Create a projection
            logits = self.output_projection(output)

            res_logits[i, :] = logits

            preds[i] = logits.argmax(dim=-1)

            # Now update the context with the prediction
            pred_emb = self.byte_embedder(preds[i].unsqueeze(0))

            ctx_payload[O_BYTE_CTX_LEN + i, :] = pred_emb

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
def train_conv(
    model: OnlinePacketPredictor,
    optimizer,
    criterion,
    conv_df: PacketDataset,
    max_conv_len: float,
) -> ConvResults:
    packet_ctx = []
    packet_losses = []
    packet_accs = []
    packet_lens = []
    packet_cnt = 0

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

                diffopt.step(loss)

                # Store results only periodically to save memory
                packet_losses.append(loss)
                correct = (results.preds == target_payload).sum().item()
                packet_accs.append(correct / len(packet.payload))
                packet_lens.append(len(packet.payload))

                packet_cnt += 1

                del target_payload

                if packet_cnt >= max_conv_len:
                    break

            assert (
                len(packet_ctx) <= O_PACKET_CTX_LEN
            ), f"The packet context length {len(packet_ctx)} is not less than or equal to {O_PACKET_CTX_LEN}"

            packet_ctx.append(packet)
            packet_ctx = packet_ctx[-O_PACKET_CTX_LEN:]

    return ConvResults(
        packet_losses, packet_accs, packet_lens=packet_lens, mem_usage=get_used_memory()
    )


def inference_conv(
    model: OnlinePacketPredictor,
    optimizer,
    criterion,
    conv_df: PacketDataset,
    max_conv_len: float,
) -> ConvResults:
    """
    @Description: Run the conversation in inference mode where we do not preserve losses or
    gradients for support training

    @Notes:

    """
    packet_ctx = []
    packet_losses = []
    packet_accs = []
    packet_lens = []
    packet_cnt = 0

    for packet in conv_df:
        if len(packet.payload) == 0:
            continue

        if len(packet_ctx) == O_PACKET_CTX_LEN:
            target_payload = torch.tensor(
                packet.payload, dtype=torch.long, device=DEVICE
            )

            results = model.forward(packet_ctx, packet)
            loss = criterion(
                results.logits.view(-1, VOCAB_DIM), target_payload.view(-1)
            )

            # TBD!! how should the training cut off be done?
            if packet_cnt < O_NO_TRAIN_N_PACKETS:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            packet_losses.append(loss.detach())
            correct = (results.preds == target_payload).sum().item()
            packet_accs.append(correct / len(packet.payload))
            packet_lens.append(len(packet.payload))

            packet_cnt += 1

            del target_payload

            if packet_cnt >= max_conv_len:
                break

        packet_ctx.append(packet)
        packet_ctx = packet_ctx[-O_PACKET_CTX_LEN:]

    return ConvResults(
        packet_losses, packet_accs, packet_lens=packet_lens, mem_usage=get_used_memory()
    )


def train_model(csv_dir: str, model=None, num_epochs=O_NUM_EPOCHS):
    """Train the packet generator model"""
    packet_generator = PacketItGenerator(csv_dir)
    data_split_dict = packet_generator.generate_conv_loaders()

    # Initialize the model
    metadata_embeder = MetadataEmbedder(
        data_split_dict["train"][0].cat_dims, data_split_dict["train"][0].num_dims
    )
    model = (
        OnlinePacketPredictor(metadata_embedder=metadata_embeder)
        if not isinstance(model, OnlinePacketPredictor)
        else model
    )

    # Create optimizer for inference time parameters
    inference_params, support_params = model.get_component_groups()
    inference_optim = torch.optim.Adam(
        inference_params, lr=O_INFERENCE_LR, weight_decay=O_WEIGHT_DECAY
    )
    support_optim = torch.optim.AdamW(
        support_params, lr=O_LR, weight_decay=O_WEIGHT_DECAY
    )

    # criterion = LabelSmoothingCrossEntropy(
    #     smoothing=O_SMOOTHING
    # )  # Label smoothing for robustness
    # criterion = nn.CrossEntropyLoss()
    criterion = FocusLoss(gamma=O_GAMMA, alpha=O_ALPHA)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        support_optim,
        max_lr=O_MAX_LR,
        total_steps=num_epochs * len(data_split_dict["train"]),
        pct_start=0.1,
    )

    print(sum(p.numel() for p in model.parameters()))

    model.to(device=DEVICE)

    # Metrics #
    model_metrics = dict()
    for mode in data_split_dict.keys():
        model_metrics[mode] = ModelMetrics()

    for epoch in range(num_epochs):
        # Training phase
        packet_generator.n_conv_packets = (
            O_MAX_CONV_LEN  # Permit the max number of conv packets
        )

        # Gradually decrease temperature and teacher forcing
        temperature = max(0.5, 1.0 - epoch / num_epochs)
        max_conv_len = int(
            O_MAX_CONV_LEN * (epoch + 1) / num_epochs
        )  # TBD! should I add NL scheduling?
        model.temperature = temperature

        for mode, conv_dfs in data_split_dict.items():
            print(mode)
            mode_results = EpochResults()
            mode_results.max_conv_len = max_conv_len

            for conv_df in conv_dfs:

                start_time = time.time()
                # Reset the model for the new conversation
                model.train_mode_micro_only()
                model.conv_reset()

                conv_results = (
                    train_conv(
                        model,
                        inference_optim,
                        criterion,
                        conv_df,
                        max_conv_len,
                    )
                    if mode == "train"
                    else inference_conv(
                        model, inference_optim, criterion, conv_df, max_conv_len
                    )
                )

                if len(conv_results.packet_losses) > 0:
                    # TBD!! what is a good loss function for this issue?
                    conv_loss = conversation_tradjectory_loss(
                        torch.stack(conv_results.packet_losses)
                    )
                    # conv_loss = conversation_trajectory_loss_simple(
                    #     torch.stack(conv_results.packet_losses)
                    # )

                    # Loss and feedback stuff
                    if mode == "train":
                        model.train_mode_support_only()
                        support_optim.zero_grad()
                        conv_loss.backward()
                        support_optim.step()
                        scheduler.step()

                    end_time = time.time()

                    # Ensure the memory is freed
                    conv_results.conv_loss = conv_loss.item()
                    detached_packet_losses = [
                        l.clone().detach().to(device="cpu")
                        for l in conv_results.packet_losses
                    ]
                    del conv_results.packet_losses
                    del conv_loss
                    conv_results.packet_losses = detached_packet_losses
                    conv_results.tot_time = end_time - start_time

                    mode_results.conv_results.append(conv_results)

                    # Print results
                    if DEBUG_MODE:
                        print_update(
                            mode=mode,
                            epoch_number=epoch,
                            max_num_packets=max_conv_len,
                            conv_results=str(conv_results),
                            epoch_mode_results=str(mode_results),
                        )

                torch.cuda.empty_cache()

            if mode_results.n_convs > 0:
                model_metrics[mode].epoch_results.append(mode_results)

        # plot the losses and accuracies
        if DEBUG_MODE:
            process_model_metrics(model_metrics)

        # Regenerate the loaders for the next epoch
        data_split_dict = packet_generator.generate_conv_loaders()

    # Training is complete, save the model, metadata, and results
    process_model_metrics(model_metrics)

    # Create a new directory
    i = 0
    model_save_dir = os.path.join(O_SAVE_DIR, f"online_model_{i}")
    while os.path.exists(model_save_dir):
        i += 1
        model_save_dir = os.path.join(O_SAVE_DIR, f"online_model_{i}")

    os.makedirs(model_save_dir)

    metadata = {
        "device": DEVICE,
        "seed": SEED,
    }
    # write the model with metrics and metadata to the directory
    pkl_write_model(model, model_metrics, model_save_dir, metadata=metadata)


if __name__ == "__main__":

    csv_dir = "test_data"
    # csv_dir = (
    #     "../datasets/mqtt-data/kaggle_mqtt_set/Data/PCAP/legit_cap_split/small_sample"
    # )
    # csv_dir = (
    #     "datasets/mqtt-data/kaggle_mqtt_set/Data/PCAP/legit_cap_split/small_sample"
    # )

    # Load an existing model
    # model, metrics, metadata = pkl_read_model(O_SAVE_DIR + "/online_model_0")

    with torch.backends.cudnn.flags(enabled=False):
        train_model(csv_dir=csv_dir, model=None)
