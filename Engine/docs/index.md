# OneBitLLM Documentation

Welcome to OneBitLLM - an experimental Rust engine for 1-bit and ternary language-model components.

## Quick Start

**New to OneBitLLM?** Start here:
- [Getting Started](./GETTING_STARTED.md) - Set up and run your first model

## Documentation

### Core Concepts
- [Architecture](./ARCHITECTURE.md) - System design and component overview
- [API Reference](./API_REFERENCE.md) - Detailed API documentation

### Development
- [Development Guide](./DEVELOPMENT.md) - Build, test, and contribute

## Project Overview

OneBitLLM provides three main components:

### 1. Core Library (`onebitllm-core`)
The computational foundation with:
- **1-bit quantization**: Convert weights to ternary values (-1, 0, +1)
- **Tensor operations**: Efficient multi-dimensional array operations
- **Neural networks**: Transformer blocks (attention, MLP, embeddings)
- **Training & inference primitives**: Building blocks for future end-to-end workflows
- **Automatic differentiation**: Gradient computation with STE
- **Current limit**: CPU backend only; no ROCm/HIP backend is wired yet

**Documentation**: [crates/onebitllm-core/README.md](../crates/onebitllm-core/README.md)

### 2. CLI (`onebitllm-cli`)
User-friendly command-line interface with:
- `train` - Validate Rust training inputs, then fail explicitly until training is wired
- `quantize` - Validate quantization settings, then fail explicitly until OBM export is wired
- `generate` - Validate generation settings, then fail explicitly until model loading/generation is wired

**Documentation**: [crates/onebitllm-cli/README.md](../crates/onebitllm-cli/README.md)

### 3. Python Bindings (`onebitllm-python`)
Rust-hosted Python bindings for:
- Configuration management
- Tokenization
- Model metadata/loading from OBM
- Explicit unimplemented errors for generation/save paths that are not finished yet

**Documentation**: [crates/onebitllm-python/README.md](../crates/onebitllm-python/README.md)

## Key Features

- **Extreme Quantization**: 32× model compression with 1-bit weights
- **Packed Weight Kernels**: Packed ternary operations in the Rust core
- **Flexible Architecture**: Multiple positional encodings, activations
- **Explicit Failure Modes**: Placeholder CLI/binding flows fail clearly instead of writing fake outputs
- **Multi-Language**: Rust core + Python bindings

## Installation

```bash
# Clone and build
cd OneBitLLM/Engine
cargo build --release

# Or install Python bindings
cd crates/onebitllm-python
pip install maturin
maturin develop --release
```

## Quick Example

### Rust

```rust
use onebitllm_core::nn::*;

let config = AttentionConfig {
    hidden_dim: 768,
    num_heads: 12,
    // ...
};

let attention = Attention::new(&config)?;
let output = attention.forward(&input)?;
```

### CLI

```bash
onebitllm --help
```

### Python

```python
import onebitllm

print(onebitllm.version())
```

## Workspace Structure

```
Engine/
├── crates/
│   ├── onebitllm-core/    # Core computational library
│   ├── onebitllm-cli/     # Command-line interface
│   └── onebitllm-python/  # Python bindings
├── docs/                  # Documentation (this folder)
├── Cargo.toml            # Workspace manifest
└── README.md             # Main README
```

## Common Tasks

| Task | Reference |
|------|-----------|
| Build project | `cargo build --release` |
| Run tests | `cargo test --all` |
| Run benchmarks | `cargo bench -p onebitllm-core` |
| View API docs | `cargo doc --no-deps --open` |
| Train a model | [CLI Guide](./GETTING_STARTED.md#step-5-run-your-first-model) |
| Use in Python | [Python Guide](../crates/onebitllm-python/README.md) |

## Performance

1-bit quantization achieves:
- **32× weight compression**: FP32 → 1-bit storage
- **~2-5× slower inference**: Specialized ternary operations vs standard
- **32× smaller models**: Fit in memory with tiny footprint

## Contributing

See [DEVELOPMENT.md](./DEVELOPMENT.md) for:
- Setting up development environment
- Running tests and benchmarks
- Code quality guidelines
- Adding new features

## Resources

- **Full Workspace README**: [README.md](../README.md)
- **Architecture Deep Dive**: [ARCHITECTURE.md](./ARCHITECTURE.md)
- **API Reference**: [API_REFERENCE.md](./API_REFERENCE.md)
- **Cargo Documentation**: `cargo doc --no-deps --open`

## License

This repository is source-available under the custom
[OneBitLLM Research Review Only License 1.0](../LICENSE). It is not open
source. See the project root for the full terms. The project may be relicensed
under MIT later, but the current repository terms remain in effect until that
relicense is actually published.

## Support

For issues, questions, or contributions:
1. Check existing documentation
2. Search closed issues
3. Open a new issue with details
4. Request written permission before preparing patches or pull requests

---

**Ready to get started?** → [Getting Started Guide](./GETTING_STARTED.md)
