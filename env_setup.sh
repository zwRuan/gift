# !/bin/bash
set -e

# docker: runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv uv_gift --python 3.12

source uv_gift/bin/activate

uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
