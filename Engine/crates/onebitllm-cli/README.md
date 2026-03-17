# onebitllm-cli

Command-line interface for the Rust OneBitLLM engine.

## Overview

`onebitllm-cli` exposes the current Rust command surface. At the moment it can validate configs, paths, and sampling/quantization settings, but the end-to-end training, quantization-export, and generation pipelines are still being wired up.

## Installation

Build from source:

```bash
cargo build --release -p onebitllm-cli
./target/release/onebitllm --help
```

Or run directly:

```bash
cargo run --release --bin onebitllm -- --help
```

## Current Status

- `onebitllm train`: validates config/data inputs, then exits with a clear unimplemented error
- `onebitllm quantize`: validates quantization settings and input path, then exits with a clear unimplemented error
- `onebitllm generate`: validates model path and generation settings, then exits with a clear unimplemented error
- No `--device` flag or GPU backend is implemented in the current CLI

## Commands

### Overview

```bash
onebitllm --help
```

### Train

Validate training inputs for the Rust engine.

```bash
onebitllm train \
  --config config.json \
  --data dataset/train.txt \
  --output output-dir \
  --epochs 10 \
  --batch-size 32
```

**Options**:
- `--config <FILE>`: JSON model configuration
- `--data <FILE>`: Training data path
- `--output <DIR>`: Output directory
- `--epochs <N>`: Number of training epochs
- `--batch-size <N>`: Batch size
- `--lr <LR>`: Learning rate
- `--warmup-steps <N>`: Warmup steps
- `--save-every <N>`: Save interval in steps
- `--seed <SEED>`: Random seed
- Current behavior: validates inputs, then returns an unimplemented error without writing checkpoints

### Quantize

Validate post-training quantization settings.

```bash
onebitllm quantize \
  --input pretrained.safetensors \
  --output quantized.obm \
  --granularity per-tensor
```

**Options**:
- `--input <FILE>`: Input checkpoint or tensor file
- `--output <FILE>`: Quantized model save path
- `--granularity <GRAN>`: `per-tensor`, `per-channel`, or `per-group`
- `--group-size <N>`: Group size for group-wise quantization (default: 128)
- Current behavior: validates inputs, then returns an unimplemented error without writing an OBM file

### Generate

Validate generation settings for a model path.

```bash
onebitllm generate \
  --model quantized.obm \
  --prompt "Hello, the meaning of life is" \
  --max-tokens 50
```

**Options**:
- `--model <FILE>`: Path to model checkpoint/container
- `--prompt <TEXT>`: Input prompt
- `--max-tokens <N>`: Maximum tokens to generate
- `--temperature <T>`: Sampling temperature
- `--top-k <K>`: Top-k sampling cutoff
- `--top-p <P>`: Nucleus sampling cutoff
- `--repetition-penalty <P>`: Repetition penalty
- Current behavior: validates inputs, then returns an unimplemented error without generating text

## Configuration Files

### Model Configuration (JSON)

The current CLI build reads model configs through the Rust JSON loader. YAML is
not enabled here.

```json
{
  "architecture": "bitnet-b1.58",
  "hidden_size": 768,
  "num_layers": 12,
  "num_attention_heads": 12,
  "num_kv_heads": 12,
  "intermediate_size": 2048,
  "vocab_size": 32000,
  "max_seq_len": 2048,
  "activation": "silu"
}
```

### Training Data

`onebitllm train` currently validates that the `--data` path exists, but it does
not yet tokenize or stream the dataset. Use a real path so the command can
validate inputs before it returns its explicit unimplemented error.

## Examples

### Validate a Training Request

```bash
onebitllm train \
  --config model.json \
  --data dataset.txt \
  --output out \
  --epochs 3 \
  --batch-size 2
```

The command validates the config and paths, then exits with a clear
unimplemented error. It does not create checkpoints yet.

### Validate Quantization Settings

```bash
onebitllm quantize \
  --input pretrained.safetensors \
  --output quantized.obm \
  --granularity per-group:128
```

The command validates the input file and quantization settings, then exits with
an explicit unimplemented error. It does not write an OBM file yet.

### Validate Generation Settings

```bash
onebitllm generate \
  --model quantized.obm \
  --prompt "The quick brown" \
  --max-tokens 20 \
  --temperature 0.7
```

The command validates the model path and sampling settings, then exits with an
explicit unimplemented error. It does not load a tokenizer or generate tokens yet.

## Logging

Control logging verbosity with `RUST_LOG`:

```bash
# Debug level
RUST_LOG=debug onebitllm train --config config.json --data dataset.txt

# Info level (default)
RUST_LOG=info onebitllm generate --model model.obm ...

# Only CLI errors
RUST_LOG=error onebitllm ...

# Specific module
RUST_LOG=onebitllm_core::nn=debug onebitllm train ...
```

## Performance Tips

- The current CLI is a validator/fail-fast surface, not a benchmarkable runtime
- Packed ternary hot paths live in `onebitllm-core`; CLI throughput work depends on wiring a real backend and model loader first
- No GPU, ROCm, or HIP flag exists in the current CLI

## Troubleshooting

### Expecting Generated Text or Output Files

The current commands validate arguments and then stop on purpose. If you need
real training, quantization export, or generation, those Rust paths still need
to be implemented.

### Invalid Configuration

Use JSON model configs for `train`:
```bash
onebitllm train --config config.json --data dataset.txt
```

If you pass `.yaml` or `.yml`, the CLI now rejects it explicitly.

## Dependencies

The CLI depends on:
- **onebitllm-core**: Core library
- **clap**: Command-line argument parsing
- **env_logger**: Logging setup
- **indicatif**: Progress bars
- **serde_yaml**: Configuration file parsing

## Architecture

The CLI delegates to `onebitllm-core`:

```
CLI Parser (clap)
    ↓
Argument and path validation
    ↓
Specialized Handlers (train/quantize/generate)
    ↓
Core config/quantization/inference types
    ↓
Explicit unimplemented error
```

Each command:
1. Parses arguments
2. Validates paths and settings
3. Builds the corresponding Rust-side config objects
4. Refuses to write placeholder artifacts or fake success output

## Contributing

When adding new commands:

1. Create handler in `commands/` module
2. Add to `Command` enum in `commands/mod.rs`
3. Implement error handling and logging
4. Add progress bars for long-running operations
5. Include help text with examples

See the main [README.md](../../README.md) for contribution guidelines.
