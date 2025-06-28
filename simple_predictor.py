"""
@Author: Joel Weber
@Date: 2025-06-28
@Description: This is a simple lstm model that uses a fixed conversation byte context size
along with the packet meta data to predict the next byte in the sequence. To generate an
entire next packet we auto regressivly predict the next byte until the next start of sentence
SOS token is reached.

@Notes:
    - All the features are repeated for the each byte. This does lead to a slight increase in
    model size, but saves on simplicity.
    - The cell and hidden states must be explicitly preserved throughout calls.

@TODO:
    - Create embedding for the categorical and numerical features
    - Construct a context embedder
    - Make an autoregressive training process that batches the sequences up into fixed lengths
    and preserves cell and hidden states for each batch.

"""

### Python imports ###
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Iterable
from dataclasses import dataclass

### Local imports ###
from preprocessing import load_df, split_into_conversations
from custom_datasets import ConversationByteStream, Byte, ByteWithContext
from CONSTANTS import *


# Next byte predictor
class NextBytePredictor(nn.Module):
    """
    @Description: The primary lstm model that handles embeddings, prediction, state preservation,
    and output processing

    @Notes:
        - The past bytes, meta data, and current packet bytes are all embedded equally. By design
        the lstm model will learn how to weight each input embedding to best predict the next byte
        - All the meta data is repeated for each byte. Although this is a waste we save on simplicity
        and need for something like an alternate mlp to embed the meta data separately
        - We train using the true context rather than the generated context because a single error
        in the generation can cause tragic results for the rest of the sequence. Much initial training
        data would be wasted if we train in this way.
        - Due to the nature of autoregressive systems the generation will suffer from any errors.
        Therefore we need to prioritize getting bytes exactly correct, not just almost correct.
        - The total embedding dims is [BYTE_EMBED_DIM * S_BYTE_CTX_LEN + cat_embed_dims + num_embed_dims]
    """

    def __init__(self, cat_dims: List[int], numerical_dims: int, device: str = DEVICE):
        super().__init__()

        # Used to embed a single byte or generate a tensor of embeddings for a tensor of bytes
        self.byte_embedding = nn.Embedding(INPUT_VOCAB_DIM, BYTE_EMBED_DIM)

        def get_embedding_dim(n_cats: int) -> int:
            # Google's categorical embedding formuala
            return min(MAX_CAT_EMB, round((n_cats * CAT_EMB_SCALAR) ** CAT_EMB_EXPO))

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(cat_size, get_embedding_dim(cat_size))
                for cat_size in cat_dims
            ]
        )

        self.cat_embed_dim = sum(get_embedding_dim(cat_size) for cat_size in cat_dims)

        # Since all the numerical features are already longs we can simply give them on dimension
        self.numerical_embed_dim = numerical_dims

        self.input_size = S_BYTE_CTX_LEN * (
            BYTE_EMBED_DIM + self.cat_embed_dim + self.numerical_embed_dim
        )

        # Now create the acutal LSTM for inference
        self.next_byte_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=S_HIDDEN_SIZE,
            num_layers=S_LSTM_LAYERS,
            batch_first=True,
            dropout=S_LSTM_DROPOUT,
        )

        # Now create the output projector
        self.output_projection = nn.Linear(S_HIDDEN_SIZE, OUTPUT_VOCAB_DIM)

        self.device = device
        self.to(device=device)

    def forward(
        self,
        input_batch: List[ByteWithContext],
        hidden_states: torch.Tensor | None = None,
        cell_states: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        @Description: Here we take a batch of BytesWithContext and generate the next byte in
        the sequence for each of those.

        @Notes:
            - If you are training in parallel then it is important to keep in mind that
            separtate hidden and cell states must be maintained for each batch. To truely
            train with history this must be done sequentially

        @Returns: Output logits torch.Tensor([batch_size, OUTPUT_VOCAB_SIZE])
        """

        assert all(
            [isinstance(inp, ByteWithContext) for inp in input_batch]
        ), f"All the inputs in the batch must be ByteWithContext objects"
        batch_size = len(input_batch)

        # Create the embeddings for each of the byte context objects
        embedded_inputs = torch.stack(
            [self._embed_byte_context(byte_with_ctx) for byte_with_ctx in input_batch],
            dim=1,
        ).to(self.device)

        # Now run the next byte prediction lstm
        hidden_states = (
            torch.zeros((batch_size, S_HIDDEN_SIZE), dtype=torch.long)
            if hidden_states is None
            else hidden_states
        )
        assert hidden_states.shape == (
            batch_size,
            S_HIDDEN_SIZE,
        ), f"The hidden states must have shape {(batch_size, S_HIDDEN_SIZE)}, not {hidden_states.shape}"

        cell_states = (
            torch.zeros((batch_size, S_HIDDEN_SIZE), dtype=torch.long)
            if cell_states is None
            else cell_states
        )
        assert cell_states.shape == (
            batch_size,
            S_HIDDEN_SIZE,
        ), f"The cell states must have shape {(batch_size, S_HIDDEN_SIZE)}, not {cell_states.shape}"

        byte_outputs, (hidden_states, cell_states) = self.next_byte_lstm(
            embedded_inputs, (hidden_states, cell_states)
        )

        assert isinstance(hidden_states, torch.Tensor) and isinstance(
            cell_states, torch.Tensor
        ), f"The hidden and cell states must be tensors not {type(hidden_states)}, and {type(cell_states)}"

        logits = self.output_projection(byte_outputs)

        return logits, hidden_states, cell_states

    ### Helper functions ###
    def _embed_byte(self, byte: Byte) -> torch.Tensor:
        """
        @Description: Completely embeds a single byte object that contains a byte with
        categorical and numerical meta data features into a properly sized tensor

        @Returns: torch.Tensor([BYTE_EMBED_DIM + cat_embed_dim + num_embed_dim])
        """
        cat_emb_list = list()
        for i, cat_emb_layer in enumerate(self.cat_embeddings):
            cat_emb_list.append(cat_emb_layer(byte.cat_features[i]))
        cat_embs = (
            torch.cat(cat_emb_list, dim=-1)
            if cat_emb_list
            else torch.empty(0, dtype=torch.long)
        )

        byte_emb = self.byte_embedding(byte.value)

        return torch.cat([byte_emb, cat_embs, byte.numerical_features], dim=-1)

    def _embed_byte_context(self, byte_with_ctx: ByteWithContext) -> torch.Tensor:
        """
        @Description: Completely embeds the byte with context object into the
        properly sized tensor

        @Notes:
            - This only embeds the context. The target byte does not need to be embedded

        @Returns: torch.Tensor([S_BYTE_CTX_LEN * (BYTE_EMBED_DIM + cat_embed_dim + num_embed_dim)])
        """
        return torch.cat(
            [self._embed_byte(byte) for byte in byte_with_ctx.context], dim=-1
        )


### Training and validation functions ###
def split_convs(
    conv_dfs: List[ConversationByteStream],
) -> Dict[str, List[ConversationByteStream]]:
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


def train_step(
    model: NextBytePredictor,
    batch: List[ByteWithContext],
    optimizer: torch.optim.Adam,
    criterion: nn.CrossEntropyLoss,
    hidden_states: torch.Tensor | None = None,
    cell_states: torch.Tensor | None = None,
) -> Tuple[float, float, torch.Tensor | None, torch.Tensor | None]:
    """
    @Description: Conducts a forward step on the model with the provided batch and uses that generate
    the next byte predicitons and compute the model loss.

    @Notes:
        - If the train argument is true we will run back propigation on the gradient
        - We train sequentially to use the history feature of lstm models

    @Returns:
    """
    batch_size = len(batch)

    logits = torch.zeros((batch_size, OUTPUT_VOCAB_DIM), dtype=torch.long)
    actual = torch.zeros(batch_size, dtype=torch.long)
    correct_cnt = 0

    for i, byte_with_ctx in enumerate(batch):
        # must be managed sequentially since we depend on state progression
        pred_logits, hidden_states, cell_states = model.forward(
            [byte_with_ctx], hidden_states=hidden_states, cell_states=cell_states
        )

        # Compare the predicted byte to the sequential one
        logits[i, :] = pred_logits.argmax(dim=-1)
        actual[i] = torch.tensor(byte_with_ctx.target, dtype=torch.long)

        if pred_logits.argmax(-1) == actual[i]:
            correct_cnt += 1

    # Now compute the loss and make 
    loss = criterion(logits, actual)

    # Backprop step
    loss.backward()
    optimizer.step()

    acc = correct_cnt / batch_size if batch_size else float('inf')

    return loss.item(), acc, hidden_states, cell_states

def generate_step(
    model: NextBytePredictor,
    cur_byte: ByteWithContext,
    target_seq_len: int,
    hidden_states: torch.Tensor | None = None,
    cell_states: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    @Description: Autoregressively generates a sequence of bytes
    
    @Notes: 
        - We keep the categorical and numerical features the same for the predictions
        - We ignore the target byte given in the ByteWithContext it is really only 
    
    @Returns: Tuple(logits, hidden, cell)
    """
    context = cur_byte.context

    logits = torch.zeros((target_seq_len, OUTPUT_VOCAB_DIM), dtype=torch.long)

    for i in range(target_seq_len):
        # must be managed sequentially since we depend on state progression
        pred_logits, hidden_states, cell_states = model.forward(
            [cur_byte], hidden_states=hidden_states, cell_states=cell_states
        )

        # Compare the predicted byte to the sequential one
        logits[i, :] = pred_logits.argmax(dim=-1)
        # STOPPING POINT: Need to complete sequence generation

    return hidden_states, cell_states
    


@dataclass
class ConvResults:
    avg_loss: float
    avg_acc: float
    conv_loss: List[float]
    conv_acc: List[float]
    hidden_states: torch.Tensor | None
    cell_states: torch.Tensor | None


def run_conv(
    model: NextBytePredictor,
    conv_df: ConversationByteStream,
    optimizer,
    criterion,
    train: bool = True,
) -> ConvResults:
    """
    @Description: This does the batching and training for a given conversation

    @Notes:

    @Returns:
    """
    model.train()  # switches to training mode
    batch_num = 0
    conv_loss = list()
    conv_acc = list()
    hidden_states = None
    cell_states = None

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
        loss, acc, hidden_state, cell_state = forward_step(
            model,
            batch,
            optimizer,
            criterion,
            train=train,
            hidden_states=hidden_states,
            cell_states=cell_states,
        )
        batch_num += 1

        conv_loss.append(loss)
        conv_acc.append(acc)

        if DEBUG_MODE:

            # Print some helpful info about the training step
            print_update(
                batch_num=batch_num,
                batch_size=len(batch),
                loss=loss,
                acc=acc,
                avg_loss=np.mean(conv_loss),
                avg_acc=np.mean(conv_acc),
            )

    if DEBUG_MODE:
        plot_metrics(
            conv_loss, f"Conversation loss", x_label="Batches", y_label="Batch Loss"
        )
        plot_metrics(
            conv_acc,
            f"Conversation accuracy",
            x_label="Batches",
            y_label="Batch accuracy",
        )

    return ConvResults(
        np.mean(conv_loss) if len(conv_loss) > 0 else float("inf"),
        np.mean(conv_acc) if len(conv_acc) > 0 else float("inf"),
        conv_loss,
        conv_acc,
        hidden_states,
        cell_states,
    )


def model_train():
    # Get the dataset for the conversation data
    df = load_df()

    # Each packet dataset represent all the parsed packets in a single conversation
    # Each packet is split into a batch based on the byte sequence for training and comes with
    # some metadata and such as categorical and numerical features.
    splits = split_into_conversations(df)
    n_convs = len(splits)
    conv_dfs = [ConversationByteStream(df, n_convs=n_convs) for df in splits]

    if len(conv_dfs) == 0:
        print(f"Number of conversations must not be zero")
        return None, [], []

    # Get the categorical and numerical dimensions they will all be the same throughout the conversations
    cat_dims = conv_dfs[0].cat_dims
    # Adjust the cat dims conversation number to match the total number of conversations
    cat_dims[3] = len(conv_dfs)
    numerical_dim = conv_dfs[0].num_dim

    # Define the cross entropy loss model and optimizer
    byte_predictor = NextBytePredictor(cat_dims, numerical_dim, device=DEVICE)

    # Now create the optimizer and criterion
    optimizer = torch.optim.Adam(
        byte_predictor.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss(reduction="mean")

    # Metrics
    best_val_loss = float("inf")
    train_losses = list()
    val_losses = list()
    train_accs = list()
    val_accs = list()

    # Now train over n training epochs
    for epoch in range(N_EPOCHS):

        # Divide each conversation into testing, training, and validation splits
        train, validation, test = split_convs(conv_dfs).values()

        # Set the model in training mode
        byte_predictor.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        for conv_df in train:
            results = run_conv(
                byte_predictor, conv_df, optimizer, criterion, train=True
            )
            epoch_loss += results.avg_loss
            epoch_acc += results.avg_acc

        avg_train_loss = epoch_loss / len(conv_dfs)
        train_losses.append(avg_train_loss)

        avg_train_acc = epoch_acc / len(conv_dfs)
        train_accs.append(avg_train_acc)

        # now switch to validation
        ic("Switching to validation")
        byte_predictor.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for conv_df in validation:
                results = run_conv(
                    byte_predictor, conv_df, optimizer, criterion, train=False
                )
                val_loss += results.avg_loss
                val_acc += results.avg_acc
        avg_val_loss = val_loss / len(conv_dfs)
        val_losses.append(avg_val_loss)

        avg_val_acc = val_acc / len(conv_dfs)
        val_accs.append(avg_val_acc)

        # Model checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": byte_predictor.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                f"checkpoints/model_epoch_{epoch}.pt",
            )

        # Print metrics
        ic(f"Epoch {epoch+1}/{N_EPOCHS}:")
        ic(f"  Training Loss: {avg_train_loss:.4f}")
        ic(f"  Validation Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if len(val_losses) > PATIENCE:
            if all(val_losses[-PATIENCE:] > best_val_loss):
                ic("Early stopping triggered")
                break

    return mqtt_model, train_losses, val_losses, train_accs, val_accs


### Helper functions ###
def get_memory(device: str = DEVICE) -> Dict[str, float]:
    """
    @Description: gets the total memory usage in mb for the specified device

    @Notes:

    @Returns: dict of allocated and reserved memory
    """
    if device == "cuda":
        return {
            "allocated": torch.cuda.memory_allocated(device=device) / 1024**2,
            "reserved": torch.cuda.memory_reserved(device=device) / 1024**2,
            "max_reserved": torch.cuda.max_memory_reserved(device=device)
            / 1024**2,  # Peak usage
        }
    else:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "resident": memory_info.rss / 1024**2,  # Resident Set Size in MB
            "virtual": memory_info.vms / 1024**2,  # Virtual Memory Size in MB
        }


@dataclass
class BytePrediction:
    byte: int
    prob: float
    prob_delta: float  # difference between best and second best candidate
    median_prob: float
    mean_prob: float
    std_prob: float


def get_preds(logits: torch.Tensor) -> List[BytePrediction]:
    """
    @Description: Gets the predicted bytes for logits of shape
    [batch_size, byte_vocab] along with the following metrics
        pred: actual value
        prob: prediciton probability


    @Notes:

    @Returns:
    """
    if len(logits.shape) == 1:
        logits = logits.reshape(1, -1)

    n_bytes, byte_dims = logits.shape
    assert (
        byte_dims == BYTE_VOCAB_DIM
    ), f"The logits byte dimmensions {byte_dims} != BYTE_EMBED_DIMS {BYTE_EMBED_DIM}"

    preds = list()

    for byte_pdf in logits:
        pred = int(byte_pdf.argmax(dim=-1))
        prob = float(byte_pdf[pred])
        mask = torch.ones_like(byte_pdf, dtype=torch.bool)
        mask[pred] = False
        pred2 = int(byte_pdf[mask].argmax(dim=-1))
        prob2 = float(byte_pdf[pred2])

        preds.append(
            BytePrediction(
                pred,
                prob,
                prob - prob2,
                float(byte_pdf.median(dim=-1).values),
                float(byte_pdf.mean(dim=-1, dtype=torch.float32)),
                float(byte_pdf.std(dim=-1)),
            )
        )

    return preds


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    @Description: A percentage indication of how accurate the model is making
    predictions.

    @Notes:
        - acc = 0 => perfect predcition
        - 0.1 < acc < 0.2 => good range
        - acc > 0.3 => poor preformance

    @Args:
        logits: torch.Tensor.shape = [n_bytes, byte_dims]
        targets: torch.Tensor.shape = [n_bytes]

    @Returns: accuracy percentage
    """
    total = targets.size(0)

    if total == 0:
        return 1.0

    preds = get_preds(logits)
    predictions = logits.argmax(-1)
    correct = (predictions == targets).sum().item()

    return (correct / total) * 100


def print_update(batch_num: int, **kwargs):
    print(f"\n Train step: {batch_num}")
    for key, val in kwargs.items():
        print(f"    {key}: {val}")

    mem_stats = get_memory()
    for key, value in mem_stats.items():
        print(f"    {key} memory: {value} MB")

    print()


def plot_metrics(
    loss_data: List[float] | np.ndarray,
    title: str | None = None,
    x_label: str = "Batch",
    y_label: str = "Batch loss",
):
    """
    @Description: Creates a line plot of the loss over time

    @Notes:

    @Returns:
    """
    plt.plot(loss_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title is None:
        title = f"{y_label} vs {x_label}"
    plt.title(title)
    plt.show()


def load_model_checkpoint(
    checkpoint_path: str, cat_dims: list, numerical_dims: int
) -> Tuple[HierarchicalMQTTModel, torch.optim.Adam, int, List[float], List[float]]:
    """
    @Description: This loads the parameters and weights of a HieararhicalMQTTModel
    from a file so that it can be used for inference or further training

    @Notes:

    @Returns:
    """
    assert os.path.exists(
        checkpoint_path
    ), f"The checkpoint path {checkpoint_path} does not exist"

    # initialize the model and optimizer
    model = HierarchicalMQTTModel(
        categorical_dims=cat_dims, numerical_dim=numerical_dims
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Now get the model from the saved checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Now restore the state
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    train_loss = checkpoint["train_loss"]
    val_loss = checkpoint["val_loss"]

    return model, optimizer, epoch, train_loss, val_loss


if __name__ == "__main__":
    # The only thing to do is call the model_train function
    mqtt_model, train_losses, validation_losses, train_accs, validation_accs = (
        model_train()
    )

    plot_metrics(
        train_losses, "Overall training losses", x_label="Epoch", y_label="Avg loss"
    )
    plot_metrics(
        validation_losses,
        "Overall validation losses",
        x_label="Epoch",
        y_label="Avg loss",
    )

    plot_metrics(
        train_acces, "Overall training accs", x_label="Epoch", y_label="Avg acc"
    )
    plot_metrics(
        validation_accs,
        "Overall validation accs",
        x_label="Epoch",
        y_label="Avg acc",
    )
