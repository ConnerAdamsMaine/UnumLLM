#!/usr/bin/env bash

set -euo pipefail

RUST_LOG="${RUST_LOG:-info}" .venv/bin/python train.py \
  --engine ./Engine/target/release/onebitllm \
  --data /data/redpajama/train \
  --output ./output/redpajama-bigram \
  --config-out ./output/redpajama-bigram/model_config.json \
  --architecture bigram \
  --hidden-size 256 \
  --num-layers 1 \
  --num-attention-heads 1 \
  --num-kv-heads 1 \
  --intermediate-size 256 \
  --vocab-size 256 \
  --max-seq-len 256 \
  --activation silu \
  --positional-encoding rope \
  --rope-theta 10000 \
  --no-use-bias \
  --quant-group-size 0 \
  --weight-format ternary \
  --training-weight-format ternary \
  --epochs 3 \
  --batch-size 256 \
  --lr 5e-4 \
  --weight-decay 0.01 \
  --warmup-steps 500 \
  --max-steps 50000 \
  --save-every 1000 \
  --log-every 50 \
  --seed 42 \
  --train-weight-format ternary \
  --save-weight-format ternary \
  --eval-data /data/redpajama/valid
