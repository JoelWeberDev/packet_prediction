#!/bin/bash

src_dir=$1

if [[ -d $src_dir ]]; then 

    for fpath in $src_dir/*; do
        fname="${fpath##*/}"
        name="${fname%%.pcap*}"
        num="${fname##*\.pcap}"
        if [[ -z $num ]]; then 
            num="0" 
        fi
        new_path="$src_dir/$name-$num.pcap"
        echo $new_path
        mv $fpath $new_path
    done

fi