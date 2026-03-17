# OneBitLLM Engine

A Rust engine for 1-bit/ternary language-model components, packed tensor math, and model container formats.

## Overview

OneBitLLM currently provides the core pieces for a Rust-first runtime, but it is not yet a complete end-to-end trainer or accelerator backend. Today the engine provides:

- **Core Engine** (`onebitllm-core`): The computational backbone with quantization, neural networks, and inference
- **CLI Tool** (`onebitllm-cli`): command surface that validates arguments and exposes current implementation status
- **Python Bindings** (`onebitllm-python`): configuration/model container bindings around the Rust engine

## Current Status

- The only working backend in-tree today is CPU.
- Packed ternary inference paths are implemented in the core tensor/linear layers.
- CLI `train`, `quantize`, and `generate` commands validate inputs but still fail fast because full pipelines are not wired yet.
- No ROCm/HIP backend exists yet, so MI300X support still requires a real accelerator implementation.

## Project Structure

```
Engine/
├── crates/
│   ├── onebitllm-core/      # Core library (quantization, neural networks, training)
│   ├── onebitllm-cli/       # Command-line interface
│   └── onebitllm-python/    # Python bindings
├── Cargo.toml               # Workspace configuration
└── README.md                # This file
```

## Features

### 1-Bit Quantization
- Ternary weight representation (-1, 0, +1)
- Bit-packing for efficient storage
- Learnable scales per granularity level
- Straight-through estimator (STE) for gradient computation

### Neural Network Modules
- Quantized linear layers
- Multi-head attention with rotary/ALiBi positional embeddings
- MLP blocks with configurable activation functions
- Embedding and normalization layers (RMSNorm, LayerNorm)

### Training & Inference Building Blocks
- Automatic differentiation with backpropagation
- Optimizers (SGD, Adam, AdamW)
- Packed-weight inference helpers for quantized tensor operations
- Support for HuggingFace tokenizers

### I/O & Model Management
- SafeTensors format support
- YAML configuration files
- Model serialization and loading

## Getting Started

### Build Requirements
- Rust 1.70+
- Python 3.8+ (for Python bindings)

### Build from Source

```bash
# Build all crates
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

### Using the CLI

```bash
# View available commands
cargo run --bin onebitllm -- --help

# See specific command help
cargo run --bin onebitllm -- <COMMAND> --help
```

### Python Integration

```python
import onebitllm

print(onebitllm.version())
```

## Crate Documentation

### [onebitllm-core](crates/onebitllm-core/README.md)
The core computational library containing:
- Quantization algorithms
- Tensor operations and broadcasting
- Neural network layers
- Training and inference loops
- I/O operations

### [onebitllm-cli](crates/onebitllm-cli/README.md)
Command-line interface for:
- Argument/config validation
- Current Rust pipeline status reporting
- Future model training, quantization, and inference entry points

### [onebitllm-python](crates/onebitllm-python/README.md)
Python bindings providing:
- Model loading and inference
- Configuration management
- Tokenization utilities
- Easy integration with Python workflows

## Documentation

Complete documentation is available in the [docs/](docs/) directory:
- **[Getting Started](docs/GETTING_STARTED.md)** - Setup and first model
- **[Architecture](docs/ARCHITECTURE.md)** - System design overview
- **[API Reference](docs/API_REFERENCE.md)** - Detailed API docs
- **[Development Guide](docs/DEVELOPMENT.md)** - Build and contribute

Key concepts:
- **Tensors**: Core multi-dimensional arrays with shape and layout information
- **Quantization**: Conversion of float values to ternary representations
- **Modules**: Reusable neural network building blocks
- **Graph**: Automatic differentiation graph for training
- **Backend**: Computational backend with SIMD support

## Development

### Testing
```bash
cargo test --all
cargo test --all --lib
```

### Benchmarking
```bash
cargo bench -p onebitllm-core
```

### Code Quality
```bash
# Format code
cargo fmt --all

# Check for issues
cargo clippy --all
```

## Performance

Benchmarks are available in the `benches/` directory:
- `bitpack_bench`: Bit-packing operations
- `inference_bench`: Inference performance

Run with: `cargo bench -p onebitllm-core`

## Dependencies

Key dependencies across workspace:
- **ndarray**: Multi-dimensional array operations
- **rayon**: Data parallelism
- **serde/serde_json**: Serialization
- **safetensors**: Efficient model format
- **pyo3**: Python bindings

See `Cargo.toml` for full dependency list.

## License

This repository is source-available under the custom
[OneBitLLM Research Review Only License 1.0](LICENSE). It is not open source.

The license permits only limited non-commercial review of the unmodified source
and documentation for research discussion. It does not permit use, execution,
reproduction, distribution, modification, or derivative works without prior
written permission from the copyright holder.

The current license is intended as temporary grant-period protection. The
project may be relicensed under MIT in the future, but no future license grant
is effective unless and until an MIT license is actually published for this
repository by the copyright holder.

## Contributing

External contributions and any other use of this repository require prior
written permission from the copyright holder.
