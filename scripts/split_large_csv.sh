#!/bin/bash
# Splits a large file into smaller files by line number

MAX_LC=100000

file_path="$(realpath $1)"
ext="${file_path##*.}"

if [[ -d $file_path ]]; then
    echo "$file_path is directory, cannot split"
elif [[ ! -e $file_path ]]; then
    echo "$file_path does not exist" 
elif [[ ($ext != "txt") && ($ext != "csv") ]]; then
    echo "file extension must be .txt or .csv"
else
    # Also paste the header in each file
    header="$(head -n 1 $file_path)"
    # Attempt to create a directory in the file's parent directory
    split_dir="${file_path/\.*}_split" 
    echo "Making directory: $split_dir"
    if [ ! -e $split_dir ]; then
        mkdir $split_dir
    fi
    split -l "$MAX_LC" --numeric-suffixes --additional-suffix=".$ext" "$file_path" "$split_dir/split_"

    # Now iterate through and send the header into the stare of each
    for split_file in $split_dir/*; do
        if [[ $header != "$(head -n 1 $split_file)" ]]; then
            sed -i "1i$header" $split_file
        fi
    done
fi

echo "Split $file_path into $MAX_LC sized blocks"
