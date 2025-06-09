#!/bin/bash

MAX_LC=3
test_path="$(pwd)/test.txt"

ext="${test_path##*.}"
 
sed -i "1i$ext" $test_path

head -n 10 $test_path

# for ((i=0; i<50; i+=1)); do
#     # awk "NR>=$(($i * $MAX_LC)) && NR<=$((($i + 1) * $MAX_LC))" $test_path >> "${test_path%/*}/test2.txt"
#     echo $i >> $test_path

# done
