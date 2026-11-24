#!/bin/bash
# =============================================================================
# CS-433 compute-limited LLM project: the BASELINE, and an example of an EXPERIMENT RUNNER.
#
#   tokenizer (FineWeb)  ->  pretrain d12 on FineWeb (~1.0e18 FLOPs)
#   ->  SFT on GSM8K + MBPP + TinyGSM + a bit of SmolTalk  ->  greedy decoding
#
# Total: 35% of the 3e18 FLOP budget. Measured on 1xA100-80GB: pretraining 99 min, SFT 3 min,
# evaluation 1 min (the official greedy decoding). Validation score: 8.87 (runs differ by a few points).
#
# Your submission needs a runner like this one: cp runs/baseline.sh runs/experiment.sh, then change the data config,
# mixtures and training stages to your recipe. It must rebuild your model from scratch, end to end, without manual steps.
#
#   bash runs/baseline.sh                       # 1 GPU
#   NPROC=4 bash runs/baseline.sh               # 4 GPUs on one node
#   WANDB_RUN=baseline bash runs/baseline.sh    # with wandb logging (run `wandb login` first)
# =============================================================================
set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
NPROC="${NPROC:-1}"
WANDB_RUN="${WANDB_RUN:-dummy}"
TAG="${TAG:-d12_baseline}"            # tag of the base checkpoint
SFT_TAG="${SFT_TAG:-$TAG}"             # tag of the SFT checkpoint
SKIP_PRETRAIN="${SKIP_PRETRAIN:-}"     # set to 1 to reuse an existing base checkpoint $TAG (data, tokenizer, pretraining skipped)
mkdir -p "$NANOCHAT_BASE_DIR"

# Python environment (uv)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ "$NPROC" -gt 1 ]; then LAUNCH="torchrun --standalone --nproc_per_node=$NPROC -m"; SEP="--"; else LAUNCH="python -m"; SEP=""; fi

if [ -z "$SKIP_PRETRAIN" ]; then
# -----------------------------------------------------------------------------
# Data, built with the standard pipeline (dedup + decontamination + manifests), see configs/data/baseline.json:
# pretraining: FineWeb web text. d12 at a 12:1 data:param ratio trains on ~1.3B tokens; the packing crops ~35% of
# tokens, so we need ~2B tokens of documents ~ 4 shards (~0.7B tokens and ~2GB each).
# SFT: GSM8K-train, MBPP (non-test splits), 21K TinyGSM problems, 46K SmolTalk conversations.
python -m scripts.prepare_data --config configs/data/baseline.json

# Tokenizer: the fixed one (BPE, 32K vocab, ~2B characters of fineweb), the same for every team (CPU, free)
python -m scripts.tok_train

# -----------------------------------------------------------------------------
# Pretraining: d12 (110M scaling params), context 1024, nanochat's compute-optimal 12 tokens/param
$LAUNCH scripts.base_train $SEP \
    --depth=12 --max-seq-len=1024 --window-pattern=L \
    --mixture=configs/mixtures/baseline.json \
    --num-iterations=2520 \
    --device-batch-size=32 \
    --model-tag=$TAG --run=$WANDB_RUN

fi

# -----------------------------------------------------------------------------
# SFT: math (GSM8K train, with calculator tool calls), code (MBPP train, ~20K TinyGSM Python solutions),
# and a bit of general chat
$LAUNCH scripts.chat_sft $SEP \
    --model-tag=$TAG --output-tag=$SFT_TAG \
    --mixture="gsm8k:4,mbpp:8,tinygsm:1,smoltalk:1" \
    --device-batch-size=32 \
    --val-every=0 --save-every=25 \
    --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# Evaluate on the validation suite (greedy decoding) + full budget report
$LAUNCH scripts.evaluate $SEP --model sft:$SFT_TAG --output results/${SFT_TAG}.json

# Package it like a submission (then fill in submission/data_card.json)
python -m scripts.make_submission --model sft:$SFT_TAG --runner runs/baseline.sh --out submission_baseline
