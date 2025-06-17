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

# Local imports


### global definitions ###
SRC_IP_TAG = "ip.src"
DST_IP_TAG = "ip.dst"

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

### visualization definitions ###
