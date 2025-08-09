import numpy as np
import torch

# Defines the different possibilites that we could predict
SOS = 0x100  # start of sentence token
MASK = 0x101  # for markng bytes for the model to ignore
# NULL = 0x102  # padding token
N_SPECIAL_TOKNES = 2
# OUTPUT_VOCAB_DIM = 256
VOCAB_DIM = 256 + N_SPECIAL_TOKNES

### Training parameters ###
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 31
TRAIN_VAL_TEST_PERCS = np.array([0.7, 0.3, 0])
TRAIN_VAL_TEST_PERCS /= np.sum(TRAIN_VAL_TEST_PERCS)
DEBUG_MODE = True

### preprocessing definitions ###
SRC_IP_TAG = "ip.src"
DST_IP_TAG = "ip.dst"
PAYLOAD_TAG = "tcp.payload"

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


### Online packet predictor constants ###
O_HIDDEN_SIZE = 64
O_NUM_LAYERS = 2
O_BYTE_EMB_DIM = 32
O_HIDDEN_DROPOUT = 0.1
O_PACKET_CTX_LEN = 4
O_BYTE_CTX_LEN = 4
O_DROPOUT = 0.2
O_SMOOTHING = 0.2

# Meta data parameters
O_CAT_MAX_EMB_DIM = 20
O_MD_DROPOUT = 0.2
O_MD_HIDDEN_DIM = 64

# Packet representation size
O_PACKET_REP_DIM = 128  # Experiment
O_PACKET_EMB_LAYERS = 2  # Experiment
O_PACKET_EMB_DROPOUT = 0.1

# Training parameters
O_LR = 1e-3
O_INFERENCE_LR = 5e-3
O_MAX_LR = 3e-3
O_WEIGHT_DECAY = 1e-5
O_SMOOTHING = 0.1
O_DEFAULT_TEMP = 0.9
O_MAX_CONV_LEN = 100
O_LOSS_SAMPLE_PERIOD = 5
O_NUM_EPOCHS = 7
O_SAVE_DIR = "results/online_learning"
O_RESULTS_FNAME = "training_results.pkl"
O_METADATA_FNAME = "metadata.pkl"
