# onebitllm-core

Core library for 1-bit quantized LLM training and inference.

## Overview

This crate provides the computational foundation for training and running language models with 1-bit (ternary) weight quantization. It includes tensor operations, neural network modules, quantization algorithms, and training/inference loops.

## Module Structure

### Tensor Operations (`tensor/`)
- **PackedTensor**: Efficient ternary tensor storage
- **Shape & Layout**: Multi-dimensional shape management
- **Broadcasting**: NumPy-style broadcasting rules
- **Operations**: Element-wise and reduction ops
- **SIMD** (optional): Vectorized operations

### Quantization (`quant/`)
Converts floating-point values to 1-bit representations:
- **Ternary weights**: Maps to {-1, 0, +1}
- **Bit-packing**: Stores 3 weights per byte
- **Learnable scales**: Per-group quantization parameters
- **STE**: Straight-through estimator for gradient computation

### Neural Networks (`nn/`)
Reusable transformer blocks:
- **Linear**: Quantized linear layers
- **Embedding**: Token and position embeddings
- **Attention**: Multi-head attention with flexible positional encodings
- **Norm**: RMSNorm and LayerNorm
- **Activation**: ReLU, GELU, SiLU, etc.
- **Positional**: Rotary, ALiBi, and learned position encodings
- **MLP**: Feed-forward blocks

### Automatic Differentiation (`autograd/`)
- Computational graph tracking
- Backpropagation with custom operation support
- Efficient gradient accumulation

### Optimization (`optim/`)
Weight update algorithms:
- SGD with momentum
- Adam
- AdamW

### Training (`train/`)
Training loop utilities:
- Batch processing
- Forward/backward passes
- Gradient clipping and normalization
- Learning rate scheduling
- Checkpoint management

### Inference (`infer/`)
Efficient model inference:
- KV-cache for auto-regressive generation
- Batch inference support
- Token generation strategies

### I/O (`io/`)
Model persistence:
- SafeTensors format (safe, efficient binary)
- YAML configuration files
- Checkpoint save/load

### Tokenization (`tokenizer/`)
Text ↔ token conversion:
- HuggingFace tokenizer integration
- Batch tokenization
- Token utilities

### Error Handling (`error.rs`)
Unified error type across the library.

## Usage

### Basic Model Creation

```rust
use onebitllm_core::nn::{Attention, MlpBlock, RmsNorm};
use onebitllm_core::tensor::{PackedTensor, Shape};

// Create tensor
let shape = Shape::new(vec![batch_size, seq_len, hidden_dim])?;
let tensor = PackedTensor::zeros(shape)?;

// Use neural network modules
let attention = Attention::new(config)?;
let mlp = MlpBlock::new(config)?;
let norm = RmsNorm::new(hidden_dim)?;
```

### Quantization

```rust
use onebitllm_core::quant::{QuantConfig, QuantParams};

let config = QuantConfig::new(
    granularity: QuantGranularity::PerChannel,
    scale_dtype: "float32",
);

let params = QuantParams::new(config, tensor_shape)?;
let quantized = params.quantize(&weights)?;
```

### Training

```rust
use onebitllm_core::train::Trainer;
use onebitllm_core::optim::Adam;

let optimizer = Adam::new(learning_rate);
let mut trainer = Trainer::new(model, optimizer)?;

for epoch in 0..num_epochs {
    let loss = trainer.train_batch(&input_ids, &target_ids)?;
    trainer.step()?;
}
```

### Inference

```rust
use onebitllm_core::infer::Inferencer;

let inferencer = Inferencer::new(model)?;
let next_tokens = inferencer.generate(&input_ids, max_new_tokens)?;
```

## Features

### Optional Features

Enable in `Cargo.toml`:

```toml
[dependencies]
onebitllm-core = { version = "0.1", features = ["serde", "rayon", "safetensors-io"] }
```

| Feature | Purpose |
|---------|---------|
| `std` | Standard library (default) |
| `serde` | Serialization/deserialization |
| `rayon` | Data parallelism |
| `simd` | SIMD optimizations |
| `tokenizers-hf` | HuggingFace tokenizers |
| `safetensors-io` | SafeTensors format |
| `yaml-config` | YAML configuration |

## Dependencies

- **ndarray**: N-dimensional array operations
- **rand**: Random number generation
- **bytemuck**: Type-safe bit manipulation
- **num-traits**: Numeric type traits
- **log**: Logging framework
- **(optional) rayon**: Data parallelism
- **(optional) serde/serde_json**: Serialization
- **(optional) safetensors**: Binary format
- **(optional) tokenizers**: HuggingFace tokenizers
- **(optional) serde_yaml**: YAML support

## Performance

Run benchmarks with:

```bash
cargo bench -p onebitllm-core
```

Available benchmarks:
- `bitpack_bench`: Bit-packing/unpacking performance
- `inference_bench`: Inference throughput

## Testing

```bash
# Run all tests
cargo test

# Run tests with logging
RUST_LOG=debug cargo test -- --nocapture

# Test with all features
cargo test --all-features

# Test specific feature combinations
cargo test --no-default-features --features serde
```

## Examples

See the `tests/` directory for comprehensive examples of:
- Tensor operations
- Quantization workflows
- Model creation and forward passes
- Training loops
- Inference generation

## Error Handling

All operations return `Result<T>` with `OneBitError`:

```rust
use onebitllm_core::{Result, OneBitError};

match some_operation() {
    Ok(result) => println!("Success: {:?}", result),
    Err(OneBitError::ShapeError(msg)) => println!("Shape error: {}", msg),
    Err(OneBitError::QuantizationError(msg)) => println!("Quant error: {}", msg),
    Err(e) => println!("Other error: {}", e),
}
```

## Design Principles

1. **Type Safety**: Leverage Rust's type system to prevent errors at compile time
2. **Performance**: Zero-copy operations where possible, SIMD when applicable
3. **Modularity**: Independent, composable modules
4. **Flexibility**: Support multiple quantization strategies and model architectures
5. **Testability**: Comprehensive unit and integration tests

## Contributing

When adding new features:
1. Include comprehensive tests
2. Add benchmarks for performance-critical code
3. Document public APIs with doc comments
4. Follow Rust idioms and conventions
5. Ensure no breaking changes to public API

See the main [README.md](../../README.md) for contribution guidelines.
