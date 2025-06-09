#!/bin/bash

src_path=$1

# Function to extract the pcap data into a csv file
extract_pcap() {
    local pcap_path=$1
    local out_dir=$2

    if [[ -z $pcap_path ]]; then
        echo "No pcap_path provided"
        return 1
    elif [[ "${pcap_path##*.}" != "pcap" ]]; then
        echo "expected $pcap_path to have .pcap file extension"
        return 1
    elif [[ !(-e $pcap_path) ]]; then
        echo "Provided path $pcap_path does not exist"
        return 1
    elif [[ !(-d $out_dir) ]]; then
        echo "ouptut directory $out_dir does not exist, saving csv in same dir"
        out_dir="$(dirname $pcap_path)"
    fi

    pc_fname="${pcap_path##*/}"
    csv_path="$out_dir/${pc_fname%%.pcap}.csv"
    tshark -r $pcap_path -Y mqtt -T fields -e frame.number -e frame.time -e frame.time_epoch -e frame.time_delta -e frame.len -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e tcp.stream -e mqtt.hdrflags -e mqtt.len -e tcp.payload -E header=y -E separator=";" >> $csv_path
}

if [[ !(-e $src_path) ]]; then
    echo "$src_path does not exist"
elif [[ -d $src_path ]]; then
    echo "$src_path is not a directory"
    # make an output directory
    out_dir_name="extracted_csv"
    mkdir $out_dir_name
    for pc_file in $src_path/*.pcap; do 
        extract_pcap $$pc_file $out_dir_name 
        echo "Extracted mqtt data from $pc_file"
    done

else
    extract_pcap $src_path
fi
