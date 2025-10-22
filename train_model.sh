#!/bin/bash

# This script is the "Best German ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash train_model.sh
# 2) Example launch in a tmux session (because the run takes ~4 hours):
# WANDB_RUN=nanochat-german tmux new-session -s nanochat-german -d "bash train_model.sh" \; pipe-pane -o "cat >> nanochat-german.log"

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv
export PATH=$HOME/.local/bin:$PATH

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync

# additionally install trackio
uv pip install trackio

# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# trackio setup
# If you wish to use trackio for logging (it's nice!, recommended).
# Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Download already trained tokenizer
mkdir -p $NANOCHAT_BASE_DIR/tokenizer

wget -LO $NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt https://huggingface.co/stefan-it/nanochat-german-tokenizer/resolve/main/token_bytes.pt?download=true
wget -LO $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl https://huggingface.co/stefan-it/nanochat-german-tokenizer/resolve/main/tokenizer.pkl?download=true

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
# Download pretraining data
python -m nanochat.dataset -n 240

# -----------------------------------------------------------------------------
# Base model (pretraining + evaluation, d20 model)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate