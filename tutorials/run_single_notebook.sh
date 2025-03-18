##
## SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
## SPDX-License-Identifier: Apache-2.0##

# This script runs all notebooks sequentially. It also downloads required
# weights and other data.

# Run e.g. as ./run_all_notebooks.sh -gpu 0
# By default the CPU is used

#!/bin/bash

# Default value for CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -gpu) 
            if [[ "$2" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
                CUDA_VISIBLE_DEVICES="$2"
                shift # Shift past the value
            fi
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift # Shift past the key
done
export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

# List of notebooks to be executed
notebooks=(
#    "phy/Sionna_tutorial_part1.ipynb"
#    "phy/Pulse_Shaping_Basics.ipynb"
    "rt/Introduction.ipynb"
)
# Sequentially execute all notebooks
for notebook in "${notebooks[@]}"; do

    echo -e "Compiling notebook $notebook..."

    # Download asset (if needed)
    ASSET=0
    if [[ "$notebook" == "phy/Sionna_tutorial_part4.ipynb" ]]; then
        echo "Download neural receiver weights for Sionna_tutorial_part4.ipynb"
        ASSET="phy/weights-ofdm-neuralrx"
        wget -nv --no-check-certificate "https://drive.google.com/uc?export=download&id=15txi7jAgSYeg8ylx5BAygYnywcGFw9WH" -O $ASSET
    fi
    if [[ "$notebook" == "phy/Neural_Receiver.ipynb" ]]; then
        echo "Download neural receiver weights for Neural_Receiver.ipynb"
        ASSET="phy/neural_receiver_weights"
        wget -nv --no-check-certificate "https://drive.google.com/uc?export=download&id=1W9WkWhup6H_vXx0-CojJHJatuPmHJNRF" -O $ASSET
    fi
    if [[ "$notebook" == "phy/DeepMIMO.ipynb" ]]; then
        ASSET="phy/scenarios"
        echo "Download dataset for DeepMIMO.ipynb"
        mkdir $ASSET
        wget -nv https://dl.dropboxusercontent.com/s/g667xpu1x96853e/O1_60.zip -O $ASSET/O1_60.zip
        cd $ASSET && unzip O1_60.zip && cd ..
    fi

    # Run notebook
    jupyter nbconvert --to notebook --execute --inplace $notebook

    # Remove STDERR from notebook
    python3 - <<EOF
import nbformat

# Use the Bash variable passed into Python
notebook_path = "$notebook"

# Load the notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Remove all stderr outputs
for cell in nb.cells:
    if cell.cell_type == "code" and "outputs" in cell:
        cell.outputs = [o for o in cell.outputs if o.output_type != "stream" or o.name != "stderr"]

# Save changes back to the same file
with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
EOF

    echo "Removed stderr output from $notebook..."

    # Delete asset (if needed)
    if [ -f "$ASSET" ]; then
        rm "$ASSET"
    elif [ -d "$ASSET" ]; then
        rm -R "$ASSET"
    fi

    echo -e "Done compiling notebook $notebook. \n"
done