#!/usr/bin/env bash
##
## SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
## SPDX-License-Identifier: Apache-2.0
##
##

# This script runs all notebooks and downloads required weights and other data.
# With multiple GPU indices, one notebook is executed per GPU and the next
# queued notebook is assigned as soon as a GPU becomes available.

# Run e.g. as ./run_all_notebooks.sh -gpu 0,1
# By default, notebooks are executed sequentially on the CPU.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
JUPYTER_BIN="${JUPYTER_BIN:-jupyter}"

GPU_LIST=""
CHECK_ASSETS_ONLY=0

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -gpu)
            if [[ "$#" -lt 2 || ! "$2" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
                echo "-gpu requires a comma-separated list of GPU indices"
                exit 1
            fi
            GPU_LIST="$2"
            shift # Shift past the value
            ;;
        -check-assets|--check-assets)
            CHECK_ASSETS_ONLY=1
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift # Shift past the key
done

if [ -n "$GPU_LIST" ]; then
    echo "Notebook GPU workers: $GPU_LIST"
else
    echo "Notebook worker: CPU"
fi

# List of notebooks to be executed
notebooks=(
    # PHY tutorials
    "phy/Sionna_tutorial_part1.ipynb"
    "phy/Sionna_tutorial_part2.ipynb"
    "phy/Sionna_tutorial_part3.ipynb"
    "phy/Sionna_tutorial_part4.ipynb"
    "phy/5G_Channel_Coding_Polar_vs_LDPC_Codes.ipynb"
    "phy/5G_NR_PUSCH.ipynb"
    "phy/Autoencoder.ipynb"
    "phy/Bit_Interleaved_Coded_Modulation.ipynb"
    "phy/CIR_Dataset.ipynb"
    "phy/Discover_Sionna.ipynb"
    "phy/Evolution_of_FEC.ipynb"
    "phy/Hello_World.ipynb"
    "phy/Introduction_to_Iterative_Detection_and_Decoding.ipynb"
    "phy/Link_Level_Simulations_with_RT.ipynb"
    "phy/MIMO_OFDM_Transmissions_over_CDL.ipynb"
    "phy/Neural_Receiver.ipynb"
    "phy/OFDM_MIMO_Detection.ipynb"
    "phy/Optical_Lumped_Amplification_Channel.ipynb"
    "phy/Pulse_Shaping_Basics.ipynb"
    "phy/Realistic_Multiuser_MIMO_Simulations.ipynb"
    "phy/Simple_MIMO_Simulation.ipynb"
    "phy/Superimposed_Pilots.ipynb"
    "phy/Weighted_BP_Algorithm.ipynb"
    # RT tutorials
    "rt/Diffraction.ipynb"
    "rt/Introduction.ipynb"
    "rt/Mobility.ipynb"
    "rt/Radio-Maps.ipynb"
    "rt/Scattering.ipynb"
    "rt/Scene-Edit.ipynb"
    # SYS tutorials
    "sys/End-to-End_Example.ipynb"
    "sys/HexagonalGrid.ipynb"
    "sys/LinkAdaptation.ipynb"
    "sys/PHY_Abstraction.ipynb"
    "sys/Power_Control.ipynb"
    "sys/Scheduling.ipynb"
    "sys/SYS_Meets_RT.ipynb"
)

ensure_gdown() {
    if ! "$PYTHON_BIN" -c "import gdown" >/dev/null 2>&1; then
        "$PYTHON_BIN" -m pip install --quiet gdown
    fi
}

download_drive_file() {
    local file_id="$1"
    local output="$2"

    if [ -e "$output" ]; then
        echo "Using existing asset $output"
        return
    fi

    ensure_gdown || return 1
    if ! "$PYTHON_BIN" -c \
        "import gdown, sys
try:
    downloaded = gdown.download(id=sys.argv[1], output=sys.argv[2], quiet=True)
except Exception as error:
    print(f'{type(error).__name__}: {error}'.splitlines()[0], file=sys.stderr)
    sys.exit(1)
sys.exit(0 if downloaded else 1)" \
        "$file_id" "$output"; then
        rm -f -- "$output"
        return 1
    fi
    downloaded_assets+=("$output")
}

cleanup_downloaded_assets() {
    local asset
    for asset in "${downloaded_assets[@]}"; do
        rm -rf -- "$asset"
    done
}

prepare_assets() {
    local failures=()

    echo "Checking assets required by the notebooks..."
    if ! download_drive_file \
        "1LAYC_leizwvmDnaKqRUjP5szZgJc799S" \
        "phy/weights-ofdm-neuralrx.pt"; then
        failures+=("phy/Sionna_tutorial_part4.ipynb")
    fi
    if ! download_drive_file \
        "1wjBB3U8Cp4a6VMZSrLBYJl6aG5sPoG_I" \
        "phy/neural_receiver_weights"; then
        failures+=("phy/Neural_Receiver.ipynb")
    fi
    if ! download_drive_file \
        "1GAgLXQxpqcVb1skHKLYPsWM4P-N3XqEV" \
        "phy/weights-ofdm-sip.pt"; then
        failures+=("phy/Superimposed_Pilots.ipynb")
    fi

    if [ "${#failures[@]}" -ne 0 ]; then
        echo "Assets could not be downloaded for:"
        printf '  - %s\n' "${failures[@]}"
        return 1
    fi
    echo "All required notebook assets are available."
}

remove_stderr_outputs() {
    local notebook="$1"

    "$PYTHON_BIN" - "$notebook" <<'PY'
import nbformat
import sys

notebook_path = sys.argv[1]

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code" and "outputs" in cell:
        cell.outputs = [
            output
            for output in cell.outputs
            if output.output_type != "stream" or output.name != "stderr"
        ]

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
PY
}

terminate_process_tree() {
    local pid="$1"
    local child

    while read -r child; do
        [ -n "$child" ] && terminate_process_tree "$child"
    done < <(pgrep -P "$pid" 2>/dev/null || true)
    kill -TERM "$pid" 2>/dev/null || true
}

run_notebook() (
    local notebook="$1"
    local gpu="$2"
    local worker="CPU"
    local nbconvert_pid=""

    if [ -n "$gpu" ]; then
        worker="GPU $gpu"
    fi

    stop_notebook() {
        trap - INT TERM
        if [ -n "$nbconvert_pid" ]; then
            terminate_process_tree "$nbconvert_pid"
            wait "$nbconvert_pid" 2>/dev/null || true
        fi
        exit 130
    }
    trap stop_notebook INT TERM

    echo "[$worker] Executing $notebook"
    CUDA_VISIBLE_DEVICES="$gpu" "$JUPYTER_BIN" nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=-1 \
        "$notebook" &
    nbconvert_pid=$!
    if ! wait "$nbconvert_pid"; then
        echo "[$worker] Execution failed: $notebook"
        exit 1
    fi
    nbconvert_pid=""

    if ! remove_stderr_outputs "$notebook"; then
        echo "[$worker] STDERR cleanup failed: $notebook"
        exit 1
    fi

    if ! "$PYTHON_BIN" "$SCRIPT_DIR/clean_notebook_outputs.py" "$notebook"; then
        echo "[$worker] Progress-output cleanup failed: $notebook"
        exit 1
    fi

    echo "[$worker] Finished $notebook"
)

downloaded_assets=()
trap cleanup_downloaded_assets EXIT

if ! prepare_assets; then
    exit 1
fi

if [ "$CHECK_ASSETS_ONLY" -eq 1 ]; then
    exit 0
fi

if [ -n "$GPU_LIST" ]; then
    IFS=',' read -r -a workers <<< "$GPU_LIST"
else
    workers=("")
fi

declare -A pid_to_notebook=()
declare -A pid_to_worker=()
next_notebook=0
running_jobs=0
failures=()

launch_notebook() {
    local worker="$1"
    local notebook="${notebooks[$next_notebook]}"
    local pid

    run_notebook "$notebook" "$worker" &
    pid=$!
    pid_to_notebook["$pid"]="$notebook"
    pid_to_worker["$pid"]="$worker"
    next_notebook=$((next_notebook + 1))
    running_jobs=$((running_jobs + 1))
}

stop_scheduler() {
    local pid

    trap - INT TERM
    echo "Stopping notebook workers..."
    for pid in "${!pid_to_notebook[@]}"; do
        terminate_process_tree "$pid"
    done
    wait 2>/dev/null || true
    exit 130
}
trap stop_scheduler INT TERM

for worker in "${workers[@]}"; do
    if [ "$next_notebook" -lt "${#notebooks[@]}" ]; then
        launch_notebook "$worker"
    fi
done

while [ "$running_jobs" -gt 0 ]; do
    finished_pid=""
    if wait -n -p finished_pid "${!pid_to_notebook[@]}"; then
        status=0
    else
        status=$?
    fi

    notebook="${pid_to_notebook[$finished_pid]}"
    worker="${pid_to_worker[$finished_pid]}"
    unset 'pid_to_notebook[$finished_pid]'
    unset 'pid_to_worker[$finished_pid]'
    running_jobs=$((running_jobs - 1))

    if [ "$status" -ne 0 ]; then
        failures+=("$notebook")
    fi

    if [ "$next_notebook" -lt "${#notebooks[@]}" ]; then
        launch_notebook "$worker"
    fi
done

if [ "${#failures[@]}" -ne 0 ]; then
    echo "The following notebooks failed:"
    printf '  - %s\n' "${failures[@]}"
    exit 1
fi

echo "Successfully executed all ${#notebooks[@]} notebooks."
