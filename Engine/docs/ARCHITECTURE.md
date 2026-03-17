# OneBitLLM Architecture

## System Overview

OneBitLLM is a modular Rust framework for 1-bit quantized language models. The architecture is organized into three main tiers:

1. **Core Library** (`onebitllm-core`): Low-level tensor operations and model components
2. **CLI Layer** (`onebitllm-cli`): Command-line interface for model operations
3. **Python Interface** (`onebitllm-python`): High-level Python bindings

## Core Components (onebitllm-core)

### 1. Tensor Subsystem (`tensor/`)

**Purpose**: Foundation for all numerical computations

```
tensor/
├── packed_tensor.rs    # Packed ternary tensor storage
├── ops.rs             # Element-wise and reduction operations
├── shape.rs           # Shape and stride management
├── broadcast.rs       # Broadcasting rules and operations
└── simd/              # SIMD optimization (optional)
```

**Key Types**:
- `PackedTensor`: Efficient ternary tensor representation (-1, 0, +1)
- `Shape`: N-dimensional shape with contiguity information
- Operations: broadcast, transpose, reshape, slice

### 2. Quantization Subsystem (`quant/`)

**Purpose**: Convert full-precision values to 1-bit representations

```
quant/
├── ternary.rs   # Ternary weight representation
├── bitpack.rs   # Bit-packing for storage efficiency
├── scales.rs    # Per-group quantization scales
├── ste.rs       # Straight-through estimator (gradient)
└── mod.rs       # Quantization pipeline
```

**Key Concepts**:

- **TernaryWeight**: Maps float weights to {-1, 0, +1}
- **QuantConfig**: Configurable quantization granularity (per-tensor, per-channel, per-group)
- **QuantParams**: Learnable scales and zero-points per group
- **PackedTernary**: Bit-packed storage (3 ternary weights per 8 bits)
- **STE**: Straight-Through Estimator for gradient flow during training

**Workflow**:
```
Float32 Weights → Normalize → Quantize to Ternary → Pack Bits → Store
                                        ↓
                              Learn Scales/Offsets
```

### 3. Neural Network Modules (`nn/`)

**Purpose**: Reusable building blocks for transformer models

```
nn/
├── module.rs      # Parameter wrapper
├── linear.rs      # QuantizedLinear layer (int8 → quant)
├── embedding.rs   # Token and position embeddings
├── attention.rs   # Multi-head attention with customizable PE
├── norm.rs        # RMSNorm and LayerNorm
├── activation.rs  # Activation functions (ReLU, GELU, SiLU)
├── positional.rs  # PE methods: Rotary, ALiBi, Learned
└── mlp.rs         # MLP blocks with configurable activation
```

**Module Hierarchy**:
```
Transformer Layer
├── Embedding (token + position)
├── Attention Block
│   ├── Linear (Q, K, V projections)
│   └── Attention (with PE: Rotary/ALiBi/Learned)
└── MLP Block
    ├── Linear (up-projection)
    ├── Activation
    └── Linear (down-projection)
```

### 4. Automatic Differentiation (`autograd/`)

**Purpose**: Gradient computation for training

- Computational graph tracking
- Backpropagation through all operations
- Memory-efficient gradient accumulation
- Support for custom operations via STE

### 5. Optimization (`optim/`)

**Purpose**: Weight updates during training

Supported optimizers:
- SGD (with momentum)
- Adam
- AdamW (weight decay)

### 6. Training Loop (`train/`)

**Purpose**: Coordinate training process

- Data loading and batching
- Forward and backward passes
- Gradient clipping and normalization
- Learning rate scheduling
- Checkpoint saving/loading

### 7. Inference Engine (`infer/`)

**Purpose**: Shared generation primitives in the Rust core

- KV-cache management for auto-regressive generation
- Batch inference support
- Tokenization integration
- Beam search and sampling strategies
- Current limit: the CLI and Python frontend are not fully wired into this stack yet, and there is no ROCm/HIP backend

### 8. I/O System (`io/`)

**Purpose**: Model persistence and configuration

Supported formats:
- **SafeTensors**: Safe, efficient binary format
- **YAML**: Human-readable configuration files
- **JSON**: Model metadata

Operations:
- Save/load checkpoints
- Export quantized models
- Import pretrained weights

### 9. Tokenizer Integration (`tokenizer/`)

**Purpose**: Text ↔ token conversion

- HuggingFace tokenizer support
- Custom vocabulary handling
- Token batching utilities

### 10. Error Handling (`error.rs`)

Unified error type for all operations:
```rust
pub enum OneBitError {
    QuantizationError(String),
    ShapeError(String),
    ComputeError(String),
    SerializationError(String),
    // ...
}
```

## CLI Layer (onebitllm-cli)

**Purpose**: Command-line interface for validating and eventually dispatching Rust workflows

```
cli/
├── main.rs           # CLI entry point
└── commands/
    ├── train.rs      # Training argument validation + explicit fail-fast
    ├── quantize.rs   # Quantization argument validation + explicit fail-fast
    └── generate.rs   # Generation argument validation + explicit fail-fast
```

**Current Features**:
- Structured argument parsing (clap)
- Input/path validation
- Logging
- Explicit unimplemented errors instead of fake-success placeholder output

## Python Bindings (onebitllm-python)

**Purpose**: Thin Python-facing wrapper over the Rust engine surface

```
python/
├── lib.rs           # PyO3 module initialization
├── model.rs         # Model metadata + OBM loading wrapper
├── config.rs        # ModelConfig Python binding
├── generate.rs      # Generation wrapper (currently explicit fail-fast)
└── tokenizer.rs     # Tokenizer wrapper
```

**Current Exposed APIs**:
- `ModelConfig`: Model hyperparameters
- `Model`: Config/parameter metadata and OBM loading
- `Generator`: Generation entrypoint that currently reports unimplemented runtime wiring
- `load()`: OBM-backed metadata load
- `save()`: Explicit unimplemented error for live-model persistence

## Data Flow Diagrams

### Target Training Flow

```
Raw Text Input
    ↓
[Tokenizer]
    ↓
Token IDs
    ↓
[Embedding + Positional Encoding]
    ↓
Embedded Sequence
    ↓
[Transformer Blocks] (multiple layers)
├── [Attention] with {Rotary/ALiBi/Learned} PE
└── [MLP]
    ↓
[Output Projection]
    ↓
Logits
    ↓
[Loss Computation]
    ↓
[Backpropagation] with STE for gradients
    ↓
[Optimizer Step]
    ↓
Updated Weights
```

### Target Inference Flow (Generation)

```
Prompt Text
    ↓
[Tokenizer]
    ↓
Token IDs
    ↓
Loop:
  ├── [Embedding + Positional Encoding]
  ├── [Transformer Blocks] (with KV-cache)
  ├── [Output Projection]
  ├── Sample/Greedy Select Next Token
  └── Append to Sequence
    ↓
Generated Text
```

### Quantization Flow

```
Float32 Model Weights
    ↓
[Compute Stats] (min, max, variance)
    ↓
[Learn Scales] via quantization-aware training
    ↓
[Quantize to Ternary] {-1, 0, +1}
    ↓
[Bit-Pack] (3 weights per byte)
    ↓
Compact Quantized Model
```

## Feature Flags

Compile-time features in `onebitllm-core`:

| Flag | Purpose | Dependencies |
|------|---------|--------------|
| `std` | Standard library support (default) | - |
| `serde` | Serialization/deserialization | serde, serde_json |
| `rayon` | Data parallelism | rayon |
| `simd` | SIMD optimizations | - |
| `tokenizers-hf` | HuggingFace tokenizers | tokenizers |
| `safetensors-io` | SafeTensors format support | safetensors, serde |
| `yaml-config` | YAML configuration | serde_yaml, serde |

## Performance Considerations

### Memory Efficiency
- **1-bit weights**: 32× reduction vs FP32
- **Bit-packing**: 3 ternary weights per byte
- **KV-cache**: Avoids recomputation in generation

### Compute Efficiency
- **SIMD**: Optional vectorized operations
- **Rayon**: Parallel tensor operations
- **Cache locality**: Contiguous tensor layouts

### Quantization Trade-offs
- **Accuracy**: 1-bit quantization → lower precision
- **Speed**: Specialized ternary operations
- **Training**: STE allows gradient flow through quantization

## Extension Points

To add new features:

1. **New Optimization Algorithm**: Implement in `optim/`
2. **Custom Activation**: Add to `nn/activation.rs`
3. **New Positional Encoding**: Add to `nn/positional.rs`
4. **Backend Optimization**: Add to `backend/` or `tensor/simd/`
5. **I/O Format**: Extend `io/` module
6. **CLI Command**: Add to `cli/commands/`

## Dependencies Overview

### Core Dependencies
- **ndarray**: N-dimensional arrays
- **num-traits**: Numeric traits
- **bytemuck**: Type-safe bit manipulation

### Optional Dependencies
- **rayon**: Parallelization
- **tokenizers**: HF tokenizer support
- **safetensors**: Efficient model format
- **serde/serde_json/serde_yaml**: Serialization
- **pyo3**: Python bindings

### Development
- **criterion**: Benchmarking
- **approx**: Numerical testing

## Testing Strategy

- **Unit tests**: Individual module tests
- **Integration tests**: End-to-end workflows
- **Benchmarks**: Performance tracking
- **Numerical precision**: Approximation testing for quantization
