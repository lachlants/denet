#!/bin/bash

if ! [ -e "./gdrive" ]; then
    echo "Downloading gdrive downloader"
    wget --quiet -O gdrive https://docs.google.com/uc?id=0B3X9GlR6EmbnQ0FtZmJJUXEyRTA&export=download
    wait
    chmod 777 gdrive
fi

INPUT_DIR=$1
if [ -z "$INPUT_DIR" ]; then
    INPUT_DIR="./"
fi

echo "Searching "$INPUT_DIR
FNAMES=$(find $INPUT_DIR -name "DRIVE_IDS")
for FNAME in $FNAMES; do
    echo "Processing "$FNAME
    DRIVE_IDS=($(cat $FNAME))
    for(( I=0; I<${#DRIVE_IDS[@]}; I++ )); do
	./gdrive download --path $(dirname $FNAME) ${DRIVE_IDS[$I]}
    done
done
