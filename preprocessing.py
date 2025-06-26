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
        "frame.number": int,
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


def extract_features(
    df: pd.DataFrame, inplace: bool = True
) -> Dict[str, Dict[str, Union[np.ndarray, List, str | int]]]:
    """
    @Description:
        - Here is the list of features that we will extract or create from the dataframe
            - sequences
            - mqtt message length
            - mqtt hdr flags
            - frame.time
            - frame.number
            - frame.time_delta ??
            - flow direction
            - conversation number
    @Return: Dict[str, List]
        - The dictionary contains all the afore mentioned columns as lists where each entry
        in the list corresponds to a row.
    """

    # Ensure that the data set has all the features that we need
    req_features = [
        "ip.src",
        "ip.dst",
        "mqtt.hdrflags",
        "mqtt.len",
        "tcp.payload",
        "frame.number",
        "frame.time_delta",
    ]

    for colname in req_features:
        assert colname in df.keys(), f"The column name {colname} not found in dataframe"

    ic("processing data frame ...")
    if inplace:
        df = df.copy(deep=True)

    n_rows = len(df)
    ret_vals = {
        "frame.number": {
            "dtype": "numerical",
            "values": np.array(df["frame.number"]),
            "dims": 1,
        },  # numerical
        "frame.time_delta": {
            "dtype": "numerical",
            "values": np.array(df["frame.time_delta"]),
            "dims": 1,
        },  # numerical
        "mqtt.len": {
            "dtype": "numerical",
            "values": np.array(df["mqtt.len"]),
            "dims": 1,
        },  # numerical
        "mqtt.hdrcmd": {
            "dtype": "categorical",
            "values": np.zeros(n_rows, dtype=np.uint8),
            "dims": 2**4,
        },  # first 4 bits of header byte categorical
        "mqtt.hdrflags": {
            "dtype": "categorical",
            "values": np.zeros(n_rows, dtype=np.uint8),
            "dims": 2**4,
        },  # last 4 bits of header byte categorical
        "flow.direction": {
            "dtype": "categorical",
            "values": np.zeros(n_rows, dtype=np.uint8),  # categorical
            "dims": 3,
        },  # flow direction classification based ips
        "conv.number": {
            "dtype": "categorical",
            "values": np.zeros(n_rows, dtype=np.uint8),  # categorical
            "dims": 0,  # gets incremented as we add conversations
        },  # unique number assiged to each conversation
        "mqtt.msg": {
            "dtype": "sequential",
            "values": list(),
            "dims": 256,  # volcab size of 256 byte options
        },  # Actual data within the mqtt message. Our whole objective
    }

    # Now convert add a row of conversation number where we assign each conversation between 2
    # communicating devices a specific number. There are choose(n_ips, 2) possible converation numbers
    ips = list(col_values_set(df, "ip.src").keys())

    while len(ips) > 0:
        ip1 = ips.pop()
        for ip2 in ips:
            # Get an index map of all the rows includeing the 2 ips
            ind_mask = conversation_filter(df, ip1, ip2, ret_inds=True)
            if len(df.loc[ind_mask]) > 0:
                ret_vals["conv.number"]["dims"] += 1
                ret_vals["conv.number"]["values"][ind_mask] = ret_vals["conv.number"][
                    "dims"
                ]

    # Add the directionality conversation flag to each conversation where
    # client -> client = 0 broker -> client = 1, client -> broker = 2
    ret_vals["flow.direction"]["values"][df["ip.src"] == IP_DICT["broker"]] = 1
    ret_vals["flow.direction"]["values"][df["ip.dst"] == IP_DICT["broker"]] = 2

    # Now parse the mqtt header into commads (first 4 bits) and flags (last 4 bits)
    for i, mqtt_hdr in enumerate(df["mqtt.hdrflags"]):
        # Split the header byte in half
        val = int(mqtt_hdr, 16)  # the value is in the for "0xFF"
        ret_vals["mqtt.hdrcmd"]["values"][i] = (val & 0b11110000) // 16
        ret_vals["mqtt.hdrflags"]["values"][i] = val & 0b00001111

    # Convert each data payload into a numpy array of bytes
    # this is the raw byte data of the mqtt msg. We remove the fixed header
    # which always includes a byte for the flags and then 1 - 4 bytes for the
    # msg length.
    ret_vals["mqtt.msg"]["values"] = list(
        [
            np.array(get_byte_list(byte_str)[-1 * mqtt_len :], dtype=np.uint8)
            for mqtt_len, byte_str in zip(df["mqtt.len"], df["tcp.payload"])
        ]
    )

    ic("done processing data frame ...")

    return ret_vals


def payload_deltas(trans_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    @Description: Here we obtain the deltas between the payloads between
    sequential transmissions.

    @Notes:

    @Returns:
    """
    deltas_len = max(0, len(trans_df) - 1)
    deltas = {
        "deltas": np.zeros(deltas_len, dtype=np.float32),
        "perc_deltas": np.zeros(deltas_len, dtype=np.float32),
    }

    # ensure the dataframe is sorted by frame number
    trans_df.sort_values("frame.number")

    for i in range(deltas_len):
        pkg1 = trans_df.iloc[i][PAYLOAD_TAG]
        pkg2 = trans_df.iloc[i + 1][PAYLOAD_TAG]
        delta = abs(len(pkg1) - len(pkg2))

        for v1, v2 in zip(pkg1, pkg2):
            if v1 != v2:
                delta += 1

        if max(len(pkg1), len(pkg2)) != 0:
            perc_delta = delta / max(len(pkg1), len(pkg2))
        else:
            perc_delta = 0

        deltas["deltas"][i] = delta
        deltas["perc_deltas"][i] = perc_delta

    return deltas


def load_df(print_info: bool = False) -> pd.DataFrame:
    csv_path = "test_data/legtimate_w1-0.csv"

    df = pd.read_csv(
        csv_path,
        dtype=get_column_types(),
        na_values=[""],
        keep_default_na=True,
        delimiter=";",
    )

    if print_info:
        print(df.head())
        print(df.info())
        print(df.describe())

    return df


def split_into_conversations(df: pd.DataFrame) -> List[pd.DataFrame]:
    ips = list(col_values_set(df, SRC_IP_TAG).keys())

    convs = list()

    while len(ips) > 1:
        ip1 = ips.pop()

        for ip2 in ips:
            conv_df = conversation_filter(df, ip1, ip2)
            if len(conv_df) > 0:
                convs.append(conv_df)

    return convs


if __name__ == "__main__":

    df = load_df(print_info=True)
    # ret_vals = extract_features(df)
    # for key, value_list in ret_vals.items():
    #     value = value_list["value"]
    #     if isinstance(value, list):
    #         print(f"{key}: {len(value)}")
    #     else:
    #         print(
    #             f"{key}: {value.shape}, first: {value[0]} max: {value.max()}, min: {value.min()}"
    #         )

    ### testing functions ###
    def ips_count(df):
        ips = col_values_set(df, "ip.src", sort_count=True)

        for key, item in ips.items():
            print(f"{key}: {item.count}")

    def test_conversation_filter(df):
        # get the broker and the  CO gas sensor
        conv_df = conversation_filter(df, IP_DICT["broker"], IP_DICT["co-gas"])

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

    # Considering packet deltas #
    def view_packet_deltas(df: pd.DataFrame):
        """
        @Description: This takes conversation data frame and plots the
        packet deltas that each device sends to the other device. We would like
        to know how similar each transmission is.

        @Notes:
            - We break the df into directional tranmission from one ip to another

        @Returns:
        """
        # Get all the ips in the df
        ips = list(col_values_set(df, SRC_IP_TAG).keys())

        trans_dirs = dict()

        print(len(ips))

        while len(ips) > 0:
            ip1 = ips.pop()
            for ip2 in ips:
                trans_dirs[f"{ip1} -> {ip2}"] = df[
                    (df[SRC_IP_TAG] == ip1) & (df[DST_IP_TAG] == ip2)
                ]
                trans_dirs[f"{ip2} -> {ip1}"] = df[
                    (df[SRC_IP_TAG] == ip2) & (df[DST_IP_TAG] == ip1)
                ]

        # Now go through each df and find the deltas
        cum_deltas = list()
        cum_perc_deltas = list()

        for dir, trans_df in trans_dirs.items():
            abs_deltas, perc_deltas = payload_deltas(trans_df).values()

            if abs_deltas.size != 0:
                cum_deltas += abs_deltas.tolist()
                cum_perc_deltas += perc_deltas.tolist()

                # now plot and show the histogram of absolute deltas and percent deltas
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                ax1.hist(abs_deltas, bins=10)
                ax1.set_title("Absolute deltas")
                ax1.set_xlabel("Delta value")
                ax1.set_ylabel("Frequency")

                ax2.hist(perc_deltas, bins=10)
                ax2.set_title("Percentage deltas")
                ax2.set_xlabel("Delta percentages")
                ax2.set_ylabel("Frequency")

                fig.suptitle(f"Packet payload deltas {dir}")

                plt.tight_layout()
                plt.show()

        # Now create one entire plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.hist(cum_deltas, bins=10)
        ax1.set_title("Absolute deltas")
        ax1.set_xlabel("Delta value")
        ax1.set_ylabel("Frequency")

        ax2.hist(cum_perc_deltas, bins=10)
        ax2.set_title("Percentage deltas")
        ax2.set_xlabel("Delta percentages")
        ax2.set_ylabel("Frequency")

        fig.suptitle(f"Cumulative packet deltas")

        plt.tight_layout()
        plt.show()

    # view_packet_deltas(df)
