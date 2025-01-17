#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
#!/bin/bash

# This script runs all notebooks sequentially. It also downloads required
# weights and other data.

# Run e.g. as ./run_all_notebooks.sh -gpu 0
# By default the CPU is used

# Ensure that notebook involving Sionna RT do not execute the scene preview
export SIONNA_NO_PREVIEW=1
echo "SIONNA_NO_PREVIEW is set to: $SIONNA_NO_PREVIEW"

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
    "5G_Channel_Coding_Polar_vs_LDPC_Codes.ipynb"
    "5G_NR_PUSCH.ipynb"
    "Autoencoder.ipynb"
    "Bit_Interleaved_Coded_Modulation.ipynb"
    "CIR_Dataset.ipynb"
    "DeepMIMO.ipynb"
    "dev_blog_example.ipynb"
    "Discover_Sionna.ipynb"
    "Evolution_of_FEC.ipynb"
    "Hello_World.ipynb"
    "Introduction_to_Iterative_Detection_and_Decoding.ipynb"
    "MIMO_OFDM_Transmissions_over_CDL.ipynb"
    "Neural_Receiver.ipynb"
    "OFDM_MIMO_Detection.ipynb"
    "Optical_Lumped_Amplification_Channel.ipynb"
    "Pulse_shaping_basics.ipynb"
    "Realistic_Multiuser_MIMO_Simulations.ipynb"
    "Simple_MIMO_Simulation.ipynb"
    "Sionna_Ray_Tracing_Coverage_Map.ipynb"
    "Sionna_Ray_Tracing_Diffraction.ipynb"
    "Sionna_Ray_Tracing_Introduction.ipynb"
    "Sionna_Ray_Tracing_Mobility.ipynb"
    "Sionna_Ray_Tracing_RIS.ipynb"
    "Sionna_Ray_Tracing_Scattering.ipynb"
    "Sionna_tutorial_part1.ipynb"
    "Sionna_tutorial_part2.ipynb"
    "Sionna_tutorial_part3.ipynb"
    "Sionna_tutorial_part4.ipynb"
    "Weighted_BP_Algorithm.ipynb"
)

# Sequentially execute all notebooks
for notebook in "${notebooks[@]}"; do

    # Download asset (if needed)
    ASSET=0
    if [[ "$notebook" == "Sionna_tutorial_part4.ipynb" ]]; then
        echo "Download neural receiver weights for Sionna_tutorial_part4.ipynb"
        ASSET="weights-ofdm-neuralrx"
        wget --no-check-certificate "https://drive.google.com/uc?export=download&id=15txi7jAgSYeg8ylx5BAygYnywcGFw9WH" -O $ASSET
    fi
    if [[ "$notebook" == "Neural_Receiver.ipynb" ]]; then
        echo "Download neural receiver weights for Neural_Receiver.ipynb"
        ASSET="neural_receiver_weights"
        wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1W9WkWhup6H_vXx0-CojJHJatuPmHJNRF" -O $ASSET
    fi
    if [[ "$notebook" == "DeepMIMO.ipynb" ]]; then
        ASSET="scenarios"
        echo "Download dataset for DeepMIMO.ipynb"
        mkdir $ASSET
        wget https://dl.dropboxusercontent.com/s/g667xpu1x96853e/O1_60.zip -O $ASSET/O1_60.zip
        cd $ASSET && unzip O1_60.zip && cd ..
    fi

    # Run notebook
    jupyter nbconvert --to notebook --execute --inplace "$notebook"

    # Delete asset (if needed)
    if [ -f "$ASSET" ]; then
        rm "$ASSET"
    elif [ -d "$ASSET" ]; then
        rm -R "$ASSET"
    fi
done
