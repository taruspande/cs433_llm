#!/bin/bash
# =============================================================================
# Smoke test of the whole pipeline on CPU / Apple Silicon (~10 minutes).
# Trains a tiny model on very little data: the scores will be ~0, the point is
# to check that your installation (and your code changes!) run end to end.
#
#   bash runs/smoke_cpu.sh
# =============================================================================
set -e
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_smoke}"
mkdir -p "$NANOCHAT_BASE_DIR"
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
# the CPU build of PyTorch on a laptop; on a GPU machine keep the CUDA build (the cpu and gpu extras replace each other)
if command -v nvidia-smi &> /dev/null; then uv sync --extra gpu; else uv sync --extra cpu; fi
source .venv/bin/activate

# tiny data: parts of 1 shard of FineMath and of CodeParrot, and three SFT sources
python -m scripts.prepare_data --config configs/data/smoke.json --num-workers 4
MIX="finemath:0.8,codeparrot:0.2"
python -m scripts.tok_train --smoke-test --mixture "$MIX" --max-chars=50000000 --vocab-size=8192

# check the FLOP estimates of the architecture in nanochat/gpt.py
python -m scripts.check_flops --depth 4 --max-seq-len 256

python -m scripts.base_train --depth=4 --head-dim=64 --max-seq-len=256 --window-pattern=L \
    --mixture="$MIX" --device-batch-size=8 --batch-size=16 \
    --num-iterations=100 --eval-every=50 --eval-tokens=32768 --sample-every=-1 --model-tag=smoke
python -m scripts.chat_sft --model-tag=smoke --mixture="gsm8k:0.2,mbpp:1" --max-seq-len=256 \
    --device-batch-size=8 --batch-size=16 --num-iterations=40 \
    --eval-every=20 --eval-tokens=16384 --val-every=-1
python -m scripts.evaluate --model sft:smoke --max-problems 40
python -m scripts.make_submission --model sft:smoke --runner runs/smoke_cpu.sh --out "$NANOCHAT_BASE_DIR/submission_smoke"
python -m scripts.evaluate --submission "$NANOCHAT_BASE_DIR/submission_smoke" --max-problems 20
