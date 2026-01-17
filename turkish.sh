#!/bin/bash

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
DEFAULT_NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
# Colab convenience: if Drive is mounted and user didn't override NANOCHAT_BASE_DIR,
# default to a persistent Drive directory.
if [ -z "$NANOCHAT_BASE_DIR" ] && [ -d "/content/drive/MyDrive" ]; then
    DEFAULT_NANOCHAT_BASE_DIR="/content/drive/MyDrive/nanochat-turk-cache"
fi
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$DEFAULT_NANOCHAT_BASE_DIR}"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Turkish dataset (FineWeb2 HQ Turkish shards on Hugging Face)
# We'll download up to 256 shards (or fewer if the dataset has fewer shards).
export NANOCHAT_DATASET_BASE_URL="https://huggingface.co/datasets/altaidevorg/fineweb2-hq-turkish/resolve/main"
export NANOCHAT_DATASET_DIR="$NANOCHAT_BASE_DIR/base_data_tr"
export NANOCHAT_EVAL_SUITE="tr"
export NANOCHAT_TRAIN_SUITE="tr"

# Extra Turkish dataset (train-only augmentation): EPFL FineWeb2-HQ `tur_Latn`.
# We download the first dozen parquets as additional training data.
# See files listing: https://huggingface.co/datasets/epfml/FineWeb2-HQ/tree/main/tur_Latn
export NANOCHAT_EXTRA_DATASET_BASE_URL="https://huggingface.co/datasets/epfml/FineWeb2-HQ/resolve/main/tur_Latn"
export NANOCHAT_EXTRA_DATASET_FILENAME_TEMPLATE="000_{index:05d}.parquet"
EXTRA_NUM_PARQUETS=${NANOCHAT_EXTRA_DATASET_NUM_FILES:-12}

# Determine how many shards exist (optional but avoids 404s if dataset has fewer than 256 shards).
# Falls back to 256 if the API query fails.
NUM_SHARDS_TO_DOWNLOAD=$(
python - <<'PY'
import os, re, json, sys
import requests

repo = "altaidevorg/fineweb2-hq-turkish"
want = 256
api = f"https://huggingface.co/api/datasets/{repo}"
try:
    j = requests.get(api, timeout=20).json()
    files = [s.get("rfilename","") for s in j.get("siblings", [])]
    shard_ids = []
    for f in files:
        m = re.match(r"shard_(\d+)\.parquet$", f)
        if m:
            shard_ids.append(int(m.group(1)))
    if shard_ids:
        avail = max(shard_ids) + 1
        print(min(want, avail))
    else:
        print(want)
except Exception:
    print(want)
PY
)
echo "Turkish dataset base_url: $NANOCHAT_DATASET_BASE_URL"
echo "Turkish dataset dir:      $NANOCHAT_DATASET_DIR"
echo "Shards to download:       $NUM_SHARDS_TO_DOWNLOAD"

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# Non-interactive W&B setup (Colab-friendly).
# - Run name defaults to "DSAI585-Project" unless you override WANDB_RUN.
# - If a W&B API key file exists, we export WANDB_API_KEY from it.
#   Create the file with:  echo "YOUR_KEY" > "$NANOCHAT_BASE_DIR/wandb_api_key.txt"
export WANDB_RUN="${WANDB_RUN:-DSAI585-Project}"
export WANDB_PROJECT="${WANDB_PROJECT:-DSAI585-Project}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-$NANOCHAT_BASE_DIR/wandb_api_key.txt}"
if [ -z "$WANDB_API_KEY" ] && [ -f "$WANDB_API_KEY_FILE" ]; then
  export WANDB_API_KEY="$(head -n 1 "$WANDB_API_KEY_FILE" | tr -d '\r\n')"
fi
# Make all stages go to the same project by default (easier to find on the website).
export WANDB_PROJECT_BASE="${WANDB_PROJECT_BASE:-$WANDB_PROJECT}"
export WANDB_PROJECT_MID="${WANDB_PROJECT_MID:-$WANDB_PROJECT}"
export WANDB_PROJECT_SFT="${WANDB_PROJECT_SFT:-$WANDB_PROJECT}"
export WANDB_PROJECT_RL="${WANDB_PROJECT_RL:-$WANDB_PROJECT}"
# Force online logging unless you explicitly override (prevents silent offline runs).
export WANDB_MODE="${WANDB_MODE:-online}"
# Fix W&B service connection issues in Colab/containerized environments.
# Give the wandb service process 30 seconds to start, then fall back to offline mode in code.
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-30}"
# Ensure we're logged in (non-interactive) if an API key is available.
if [ -n "$WANDB_API_KEY" ]; then
  wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
# python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
python -m nanochat.dataset -n "$NUM_SHARDS_TO_DOWNLOAD" --extra-num-files "$EXTRA_NUM_PARQUETS" &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
# NOTE: For Turkish runs we intentionally retrain every time to avoid subtle mismatches
# between tokenizer and the current dataset shards / preprocessing.
python -m scripts.tok_train --force --max-chars=2000000000 --vocab-size=65536
# evaluate the tokenizer (report compression ratio etc.)
# python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. Also, the new DataLoader wastes about 35% of tokens to cropping
# so 240 / (1 - 0.35) = 370 shards are needed.
# At ~100MB/shard, this downloads ~37GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Number of processes/GPUs to use
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

# pretrain the d4 model
# Disable English-centric evals/samples (CORE benchmark + sample prompts) for Turkish runs.
# Keep val bpb evals (they are computed on the Turkish val split).
export NANOCHAT_SAMPLE_PROMPTS='[
  "Fransa'\''nın başkenti",
  "Altının kimyasal sembolü",
  "Dün Cuma idiyse yarın hangi gün olur",
  "Sıcağın zıttı",
  "Güneş sistemindeki gezegenler şunlardır:",
  "En sevdiğim renk",
  "5*x + 3 = 13 ise x kaçtır"
]'
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
  --depth=10 --target-param-data-ratio=20 --run=$WANDB_RUN --optim adamw \
  --core-metric-every=-1 --save-every=1000
# Evaluate train/val bpb (language-agnostic, uses Turkish val shard)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
# Keep CORE eval disabled (English-only)
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Optional: Turkish identity conversations (if you have a URL to a TR JSONL in nanochat CustomJSON format).
# Format per line: [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]
# If provided, we will use it for Turkish mid-training/SFT persona shaping.
if [ -n "$NANOCHAT_IDENTITY_CONVERSATIONS_TR_URL" ]; then
  curl -L -o $NANOCHAT_BASE_DIR/identity_conversations_tr.jsonl "$NANOCHAT_IDENTITY_CONVERSATIONS_TR_URL"
  export NANOCHAT_IDENTITY_CONVERSATIONS_FILE="$NANOCHAT_BASE_DIR/identity_conversations_tr.jsonl"
fi

# run midtraining and eval the model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft and re-eval right away (should see a small bump)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
python -m scripts.chat_cli -p "Gökyüzü neden mavi?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN
# eval the RL model only on GSM8K
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
