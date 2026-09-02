#!/bin/sh
# Bootstrap a fresh Linux GPU box (Ubuntu 22.04/24.04, NVIDIA driver present)
# for JHBR5 training and self-play. Run from the JHBR5 checkout:
#
#   tools/cloud_setup.sh [venv dir]     # default ~/venv
#
# Then copy the shards (data/pack/*.rec) and run docs/HOW_TO_TRAIN.md commands
# with $VENV/bin/python, or tools/pipeline.py with "python" pointing at the venv.
set -e
VENV=${1:-$HOME/venv}
sudo apt-get update -qq && sudo apt-get install -y -qq build-essential cmake python3 python3-venv python3-dev p7zip-full
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/pip" install -q numpy pybind11 cshogi
export PATH="$VENV/bin:$PATH"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DJHBR5_ISA=avx2 -DPython3_EXECUTABLE="$VENV/bin/python"
cmake --build build -j"$(nproc)"
ctest --test-dir build -R "nnue_kernels|nnue_net|net_roundtrip" --output-on-failure
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "ready: PYTHONPATH=build $VENV/bin/python train/train.py ..."
