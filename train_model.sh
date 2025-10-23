#!/bin/bash

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# WANDB_RUN=nanochat-german tmux new-session -s nanochat-german -d "bash train_model.sh" \; pipe-pane -o "cat >> nanochat-german.log"

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

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
uv pip install trackio

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 10 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 300 is the right number here
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
python -m scripts.tok_train --max_chars=2000000000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Download evaluation data
export NANOCHAT_EVAL_DIR=$NANOCHAT_BASE_DIR/eval_bundle
export NANOCHAT_EVAL_DATA_DIR=$NANOCHAT_BASE_DIR/eval_bundle/eval_data
mkdir -p $NANOCHAT_EVAL_DIR
mkdir -p $NANOCHAT_EVAL_DATA_DIR

wget -LO $NANOCHAT_EVAL_DIR/core.yaml  https://huggingface.co/datasets/stefan-it/nanochat-german-eval-data/resolve/main/core.yaml?download=true
wget -LO $NANOCHAT_EVAL_DIR/eval_meta_data.csv  https://huggingface.co/datasets/stefan-it/nanochat-german-eval-data/resolve/main/eval_meta_data.csv?download=true

for folder in commonsense_reasoning language_understanding reading_comprehension safety world_knowledge; do
  mkdir -p $NANOCHAT_EVAL_DATA_DIR/$folder
done

wget -LO $NANOCHAT_EVAL_DATA_DIR/commonsense_reasoning/copa.jsonl  https://huggingface.co/datasets/stefan-it/nanochat-german-eval-data/resolve/main/commonsense_reasoning/copa.jsonl?download=true

wget -LO $NANOCHAT_EVAL_DATA_DIR/language_understanding/hellaswag.jsonl  https://huggingface.co/datasets/stefan-it/nanochat-german-eval-data/resolve/main/language_understanding/hellaswag.jsonl?download=true

wget -LO $NANOCHAT_EVAL_DATA_DIR/reading_comprehension/boolq.jsonl  https://huggingface.co/datasets/stefan-it/nanochat-german-eval-data/resolve/main/reading_comprehension/boolq.jsonl?download=true

wget -LO $NANOCHAT_EVAL_DATA_DIR/safety/enterprise_pii_classification.jsonl  https://huggingface.co/datasets/stefan-it/nanochat-german-eval-data/resolve/main/safety/enterprise_pii_classification.jsonl?download=true

wget -LO $NANOCHAT_EVAL_DATA_DIR/world_knowledge/mmlu.jsonl  https://huggingface.co/datasets/stefan-it/nanochat-german-eval-data/resolve/main/world_knowledge/mmlu.jsonl?download=true

# -----------------------------------------------------------------------------
# Base model (pretraining)

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. At ~100MB/shard, this downloads ~24GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# pretrain the d20 model
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --run=$WANDB_RUN
# evaluate the model on a larger chunk of train/val data and draw some samples
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
# evaluate the model on CORE tasks
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
