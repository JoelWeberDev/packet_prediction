"""
@Author: Joel Weber
@Date: 2025-06-03
@Description: This is the centralized location where all project constants
or parameters are stored. This makes for simple referencing and parameter
adjustment.

@Notes:

@TODO:
"""

# Python imports
import numpy as np
import torch
from icecream import ic

# Local imports


### global definitions ###
SRC_IP_TAG = "ip.src"
DST_IP_TAG = "ip.dst"
PAYLOAD_TAG = "tcp.payload"

SEED = 31
np.random.seed(SEED)

MAX_DF_ROWS = 100000

NUMERICS = (int, float, np.number)
ITERABLES = (
    list,
    tuple,
)

### preprocessing definitions ###
IP_DICT = {
    "broker": "10.16.100.73",
    "smoke": "192.168.0.180",
    "lock": "192.168.0.176",
    "co-gas": "192.168.0.155",
    "motion1": "192.168.0.154",
    "motion2": "192.168.0.174",
    "temp": "192.168.0.151",
    "humidity": "192.168.0.152",
    "fan": "192.168.0.178",
    "fan_spd": "192.168.0.173",
    "light": "192.168.0.150",
}

MQTT_COMMANDS = [
    "RESERVED",  # 0  - Reserved
    "CONNECT",  # 1  - Client request to connect to Server
    "CONNACK",  # 2  - Connect acknowledgment
    "PUBLISH",  # 3  - Publish message
    "PUBACK",  # 4  - Publish acknowledgment
    "PUBREC",  # 5  - Publish received (QoS 2)
    "PUBREL",  # 6  - Publish release (QoS 2)
    "PUBCOMP",  # 7  - Publish complete (QoS 2)
    "SUBSCRIBE",  # 8  - Client subscribe request
    "SUBACK",  # 9  - Subscribe acknowledgment
    "UNSUBSCRIBE",  # 10 - Unsubscribe request
    "UNSUBACK",  # 11 - Unsubscribe acknowledgment
    "PINGREQ",  # 12 - PING request
    "PINGRESP",  # 13 - PING response
    "DISCONNECT",  # 14 - Client is disconnecting
    "RESERVED",  # 15 - Reserved
]

### LSTM model meta parameters ###
# Defines the different possibilites that we could predict
SOS = 0x100  # start of sentence token
MASK = 0x101  # for markng bytes for the model to ignore
# NULL = 0x102  # padding token
N_SPECIAL_TOKNES = 2
# OUTPUT_VOCAB_DIM = 256
VOCAB_DIM = 256 + N_SPECIAL_TOKNES
# BYTE_VOCAB_DIM = 256 + N_SPECIAL_TOKNES  # number of bytes plus special tokens

# Next packet predictor parameters #
MAX_SEQ_LEN = (
    128  # Hard cap on the payload size anything beyond this will get cut short
)
CONV_CONTEXT_LEN = 10  # Number of packets in context
BYTE_EMBED_DIM = 128  # Could be reduced since this is a major source of parameter count


PACKET_ENC_DROPOUT = 0.2
METADATA_MLP_DROPOUT = 0.2


### Training parameters ###
TRAIN_VAL_TEST_PERCS = np.array([0.70, 0.3, 0])
TRAIN_VAL_TEST_PERCS /= np.sum(TRAIN_VAL_TEST_PERCS)

N_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ic(f"Using device type: {DEVICE}")
PATIENCE = 5  # minimum early stopping epochs
CHECKPOINT_DIR = "checkpoints"
DEBUG_MODE = True

### Heiarchical Model settings ###

# Conversational lstm params #
H_CONVERSATIONAL_HIDDEN_DIM = 512
H_CONVERSATIONAL_LAYERS = 3
H_CONVERSATIONAL_LSTM_DROPOUT = 0.3

# Packet encoder params #
H_PACKET_CTX = 4
H_PAYLOAD_LEN = 32
H_METADATA_HIDDEN_DIM = 64
H_METADATA_DROPOUT = 0.3
H_METADATA_OUTPUT_DIM = 32

H_PACKET_REP_DIM = 256
H_PACKET_ENC_LAYERS = 2
H_PACKET_ENC_DROPOUT = 0.3
H_PACKET_METADATA_SCALE = 0.5
H_PACKET_COMBINER_DROPOUT = 0.2

# Next packet prediction params #
H_BYTE_CONTEXT_LEN = 32  # Number of current packet bytes to include in the context
H_PACKET_ENC_LAYERS = 2
H_NEXT_PACKET_LAYERS = 2
H_NEXT_PACKET_DROPOUT = 0.2
H_TEMP = 1.0


### Simple lstm model params ###
S_BYTE_CTX_LEN = 32
S_HIDDEN_SIZE = 512
S_LSTM_LAYERS = 3
S_LSTM_DROPOUT = 0.2
S_TEMPERATURE = 1.0

### Magic numbers ###
CAT_EMB_SCALAR = 1.6
CAT_EMB_EXPO = 0.56
MAX_CAT_EMB = 50

### Primitive next byte predictor ###
P_CTX_LEN = 32
P_HIDDEN_SIZE = 128
P_NUM_LAYERS = 2
P_DROPOUT = 0.2
P_LEARNING_RATE = 1e-3
P_WEIGHT_DECAY = 1e-5
P_SMOOTHING = 0.2
P_TRAIN_INTERRUPT = 400
P_VAL_INTERRUPT = 700
P_CTX_MASK_PERC = 0.1
P_INITIAL_TEMPERATURE = 0.9


P_METADATA_DROPOUT = 0.5
P_METADATA_HIDDEN_DIM = 64
P_METADATA_OUTPUT_DIM = 32
P_METADATA_SCALE = 0.5
