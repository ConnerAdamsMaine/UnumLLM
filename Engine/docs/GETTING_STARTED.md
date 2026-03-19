# Getting Started with OneBitLLM

This guide walks you through setting up OneBitLLM and running your first model.

## Prerequisites

- **Rust 1.70+**: [Install Rust](https://rustup.rs/)
- **Python 3.8+** (optional): For Python bindings

Verify installation:
```bash
rustc --version  # Should be 1.70 or later
cargo --version
```

## Step 1: Clone and Build

```bash
# Navigate to Engine directory
cd OneBitLLM/Engine

# Build all crates
cargo build --release
```

Build times:
- First build: ~2-5 minutes (downloads dependencies)
- Incremental builds: ~10-30 seconds

## Step 2: Verify Installation

Run the CLI to confirm setup:

```bash
cargo run --release --bin onebitllm -- --version
```

You should see:
```
onebitllm 0.1.0
```

## Step 3: Create Sample Data

Create a simple training dataset. The current bigram trainer accepts plain text
files directly, and it can also flatten `.csv`, `.json`, `.jsonl`, and
`.ndjson` inputs into a byte-level corpus:

```bash
mkdir -p data
cat > data/train.txt <<EOF
The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks with multiple layers.
Transformers have revolutionized natural language processing.
Quantization reduces model size for efficient inference.
EOF
```

## Step 4: Create Model Configuration

Create a small JSON model configuration for testing:

```bash
cat > config.json <<EOF
{
  "architecture": "bigram",
  "hidden_size": 256,
  "num_layers": 1,
  "num_attention_heads": 1,
  "num_kv_heads": 1,
  "intermediate_size": 256,
  "vocab_size": 256,
  "max_seq_len": 128,
  "activation": "gelu"
}
EOF
```

## Step 5: Run a Small Training Job

### Option A: Using the CLI

Run a small training request:

```bash
cargo run --release --bin onebitllm -- train \
  --config config.json \
  --data data/train.txt \
  --output out \
  --epochs 2 \
  --batch-size 2
```

The current CLI trains and writes checkpoints for the byte-level bigram path.
`architecture` must be `bigram` and `vocab_size` must be `256` for real
training execution.

### Option B: Using Rust Library

```bash
cat > examples/simple_train.rs <<'EOF'
use onebitllm_core::nn::*;
use onebitllm_core::tensor::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OneBitLLM model initialized!");
    Ok(())
}
EOF

cargo run --release --example simple_train
```

### Option C: Using Python

Install Python bindings:

```bash
cd crates/onebitllm-python
pip install maturin
maturin develop --release
```

Create a Python script:

```python
import onebitllm

print(onebitllm.version())
```

The Python bindings currently expose configuration/tokenizer/model metadata
surfaces, but generation and live-model save paths are still intentionally
fail-fast until the Rust runtime wiring is complete.

## Step 6: Run Tests

```bash
cargo test --release
RUST_LOG=debug cargo test --release -- --nocapture
```

## Step 7: Benchmark

```bash
cargo bench -p onebitllm-core
```

## What's Next?

- **Architecture**: Read [docs/ARCHITECTURE.md](./ARCHITECTURE.md)
- **Core Library**: See [crates/onebitllm-core/README.md](../crates/onebitllm-core/README.md)
- **CLI Guide**: Check [crates/onebitllm-cli/README.md](../crates/onebitllm-cli/README.md)
- **Python API**: View [crates/onebitllm-python/README.md](../crates/onebitllm-python/README.md)

## Quick Reference

```bash
# Build
cargo build --release

# Test
cargo test --release

# Benchmark
cargo bench -p onebitllm-core

# Format code
cargo fmt --all

# Check issues
cargo clippy --all

# Generate docs
cargo doc --no-deps --open

# CLI help
cargo run --release --bin onebitllm -- --help
```

## Troubleshooting

**Build fails**: Install build tools (gcc, llvm)
```bash
# Ubuntu
sudo apt-get install build-essential

# macOS
brew install llvm
```

**Python import error**: Rebuild bindings
```bash
cd crates/onebitllm-python
maturin develop --release
```

**Out of memory**: Reduce parallelism
```bash
cargo build -j 1 --release
```
