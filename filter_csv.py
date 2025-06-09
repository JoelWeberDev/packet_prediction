"""
@Author: Joel Weber
@Date: 2025-06-03
@Description: We use this to parse, filter, and analyze the MQTT csv files.
The data set is nicely parsed into possible mqtt headings. It should be
noted that in most packets most headings are blank (having no value at all).

@Notes:
    - CSV header:
        frame.time_delta,frame.time_delta_displayed,frame.time_epoch,
        frame.time_invalid,frame.time_relative,eth.src,eth.dst,ip.src,
        ip.dst,tcp.srcport,tcp.dstport,tcp.flags,frame.cap_len,frame.len,
        frame.number,tcp.stream,tcp.analysis.initial_rtt,tcp.time_delta,
        tcp.len,tcp.window_size_value,tcp.checksum,mqtt.clientid,
        mqtt.clientid_len,mqtt.conack.flags,mqtt.conack.flags.reserved,
        mqtt.conack.flags.sp,mqtt.conack.val,mqtt.conflag.cleansess,
        mqtt.conflag.passwd,mqtt.conflag.qos,mqtt.conflag.reserved,
        mqtt.conflag.retain,mqtt.conflag.uname,mqtt.conflag.willflag,
        mqtt.conflags,mqtt.dupflag,mqtt.hdrflags,mqtt.kalive,mqtt.len,
        mqtt.msg,mqtt.msgid,mqtt.msgtype,mqtt.passwd,mqtt.passwd_len,
        mqtt.proto_len,mqtt.protoname,mqtt.qos,mqtt.retain,mqtt.sub.qos,
        mqtt.suback.qos,mqtt.topic,mqtt.topic_len,mqtt.username,
        mqtt.username_len,mqtt.ver,mqtt.willmsg,mqtt.willmsg_len,
        mqtt.willtopic,mqtt.willtopic_len,ip.proto

@TODO
    - Creation of pandas data frames
    - Simply read a csv dataset
    - Create methods to filter for the following things (return new dataframe)
        - Packets with specific value(s) for a certain header
        - Packets with non trivial values for specific header
        - Conversation filter
        - Timestamp range filter
    -

"""

# Python imports
import sys, os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from icecream import ic
from dataclasses import dataclass
from typing import Type, Union, Dict, List

# Local imports
from CONSTANTS import *
import visualization as vis


### object definitions ###
@dataclass
class UniqueValue:
    value: Union[float, str, int]
    count: int
    dtype: Type
    # Perhaps more interesting data in the future


### Filtering functons ###
def conversation_filter(
    df: pd.DataFrame, ip1: str, ip2: str, ret_inds: bool = False
) -> pd.DataFrame:
    """
    @Description: This returns a dataframe that only includes transmissions
    between the 2 devices at ip1 and ip2.

    @Notes:

    @Returns: Filtered dataframe
    """
    if df.get(SRC_IP_TAG) is None or df.get(DST_IP_TAG) is None:
        ic(f"Dataframe missing source or dest ip addresses")
        return df

    inds = ((df[SRC_IP_TAG] == ip1) & (df[DST_IP_TAG] == ip2)) | (
        (df[SRC_IP_TAG] == ip2) & (df[DST_IP_TAG] == ip1)
    )

    if ret_inds:
        return inds

    return df[inds]


def col_values_set(
    df: pd.DataFrame, colname: str, sort_count: bool = True
) -> Dict[str, UniqueValue]:
    """
    @Description: This gets a set of the uniue values in the colname
    along with counts of their occurrences

    @Notes:

    @Returns: Dict of key indexed UniqueValue objects
    """
    values = dict()

    col = df.get(colname)
    if col is not None:
        for value in col:
            if values.get(str(value)) is None:
                values[str(value)] = UniqueValue(value, 1, type(value))
            else:
                values[str(value)].count += 1

    if sort_count:
        values = dict(sorted(values.items(), key=lambda x: x[1].count, reverse=True))

    return values


### Helper functions ###
def get_byte_list(byte_str: str) -> List[int]:
    """
    @Description: This parses a hexadecimal byte string into list of uint8 values

    @Notes:
        - Every byte must occupy exactly 2 spaces in the string
        - All values are assumed to be hexadecimal

    @Returns:
    """
    assert (
        len(byte_str) % 2 == 0
    ), f"the byte string must have an even length, not {len(byte_str)}"

    parsed_bytes = list()

    for i in range(0, len(byte_str), 2):
        parsed_bytes.append(int(byte_str[i : i + 2], base=16))

    return parsed_bytes


def get_column_types() -> Dict[str, type]:
    """
    @Description: Defines the data types for each column in the CSV file

    @Returns: Dictionary mapping column names to their types
    """
    return {
        "frame.time_delta": float,
        "frame.time": str,
        "frame.time_epoch": float,
        "frame.time_relative": float,
        "eth.src": str,
        "eth.dst": str,
        "ip.src": str,
        "ip.dst": str,
        "tcp.srcport": int,
        "tcp.dstport": int,
        "frame.len": int,
        "tcp.len": int,
        "mqtt.len": int,
        "mqtt.topic_len": int,
        "mqtt.clientid": str,
        "mqtt.topic": str,
        "mqtt.msg": str,
        # Add other columns as needed
    }


if __name__ == "__main__":

    csv_path = "/home/joel/Documents/laurier/URSA/research/datasets/mqtt-data/kaggle_mqtt_set/Data/PCAP/legit_cap_split/legtimate_w1-0.csv"

    df = pd.read_csv(
        csv_path,
        dtype=get_column_types(),
        na_values=[""],
        keep_default_na=True,
        delimiter=";",
    )

    print(df.head())
    print(df.info())
    print(df.describe())

    ### data preprocessing ###
    ic("processing data frame ...")
    # Convert the time into a date time
    df["frame.time"] = pd.to_datetime(
        df["frame.time"].str.replace("EDT", "").str.strip(),
        format="%b %d, %Y %H:%M:%S.%f",
        errors="coerce",
    )
    df["frame.time"] = df["frame.time"].dt.tz_localize("America/New_York")
    # print(df["frame.time"].head())

    # Now convert add a row of conversation number where we assign each conversation between 2
    # communicating devices a specific number. There are choose(n_ips, 2) possible converation numbers
    ips = list(col_values_set(df, "ip.src").keys())
    conv_n = 1
    df["conv.n"] = np.zeros(len(df), np.int32)

    while len(ips) > 0:
        ip1 = ips.pop()
        for ip2 in ips:
            # Get an index map of all the rows includeing the 2 ips
            ind_mask = conversation_filter(df, ip1, ip2, ret_inds=True)
            if len(df.loc[ind_mask]) > 0:
                df.loc[ind_mask, "conv.n"] = conv_n
                conv_n += 1

    # Convert each data payload into a numpy array of bytes
    # this is the raw byte data of the mqtt msg. We remove the fixed header
    # which always includes a byte for the flags and then 1 - 4 bytes for the
    # msg length. Since
    df["mqtt.msg"] = list(
        [
            np.array(get_byte_list(byte_str)[-1 * mqtt_len :], dtype=np.uint8)
            for mqtt_len, byte_str in zip(df["mqtt.len"], df["tcp.payload"])
        ]
    )

    ic("done processing data frame ...")

    ### Actual model traing ###

    ### testing functions ###
    def ips_count(df):
        ips = col_values_set(df, "ip.src", sort_count=True)

        for key, item in ips.items():
            print(f"{key}: {item.count}")

    def test_conversation_filter(df):
        ip_dict = {
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

        # get the broker and the  CO gas sensor
        conv_df = conversation_filter(df, ip_dict["broker"], ip_dict["co-gas"])

        print(f"totol packets: {len(conv_df)}")

        # ips_count(conv_df)

        lens = col_values_set(conv_df, "mqtt.len", sort_count=False)
        srt_lens = dict(
            sorted(lens.items(), key=lambda x: -1 if x[0] == "nan" else int(x[0]))
        )

        for key, item in srt_lens.items():
            print(f"{key}: {item.count}")

    # test_conversation_filter(df)

    def test_bs():
        test_bytes = (
            "102300044d5154540402003c00173039303131633032316234333430376338643533303235"
        )

        parsed_bytes = get_byte_list(test_bytes)

        for i, bt in enumerate(parsed_bytes):
            print(f"{hex(bt)}, {test_bytes[2 * i : 2 * (i + 1)]}")

    # test_bs()
