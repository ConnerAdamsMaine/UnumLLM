# Development Guide

## Workspace Structure

```
Engine/
├── crates/
│   ├── onebitllm-core/      # Core computational library
│   │   ├── src/
│   │   │   ├── quant/       # 1-bit quantization
│   │   │   ├── tensor/      # Tensor operations
│   │   │   ├── nn/          # Neural network modules
│   │   │   ├── autograd/    # Automatic differentiation
│   │   │   ├── optim/       # Optimizers
│   │   │   ├── train/       # Training utilities
│   │   │   ├── infer/       # Inference engine
│   │   │   ├── io/          # Model I/O
│   │   │   ├── tokenizer/   # Tokenization
│   │   │   └── error.rs     # Error types
│   │   ├── tests/           # Integration tests
│   │   └── benches/         # Benchmarks
│   │
│   ├── onebitllm-cli/       # Command-line interface
│   │   ├── src/
│   │   │   ├── commands/    # CLI subcommands
│   │   │   └── main.rs
│   │   └── tests/
│   │
│   └── onebitllm-python/    # Python bindings
│       ├── src/             # PyO3 bindings
│       └── tests/
│
├── docs/                    # Documentation
├── Cargo.toml               # Workspace manifest
└── Cargo.lock               # Locked dependencies
```

## Building

### Full Build

```bash
cargo build --release
```

### Feature Combinations

```bash
# All features
cargo build --all-features

# Specific features
cargo build --features rayon,safetensors-io

# No defaults
cargo build --no-default-features --features std
```

### Specific Crates

```bash
cargo build -p onebitllm-core --release
cargo build -p onebitllm-cli --release
```

## Testing

### Run All Tests

```bash
cargo test --all --release
```

### Test Specific Crate

```bash
cargo test -p onebitllm-core --release
```

### Test Specific Module

```bash
cargo test quant::ternary --release
cargo test tensor::ops --release
```

### With Logging

```bash
RUST_LOG=debug cargo test --release -- --nocapture --test-threads=1
```

### Test Coverage

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin -p onebitllm-core --out Html
```

## Benchmarking

### Run Benchmarks

```bash
cargo bench -p onebitllm-core
```

### Benchmark Specific Function

```bash
cargo bench bitpack
cargo bench inference
```

### Generate HTML Reports

Criterion generates HTML reports in `target/criterion/`:

```bash
# Open report
open target/criterion/report/index.html
```

### Custom Benchmarks

Add to `crates/onebitllm-core/benches/`:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_my_feature(c: &mut Criterion) {
    c.bench_function("my_feature", |b| {
        b.iter(|| {
            // Your code here
        });
    });
}

criterion_group!(benches, benchmark_my_feature);
criterion_main!(benches);
```

## Code Quality

### Formatting

```bash
cargo fmt --all
```

### Linting

```bash
cargo clippy --all -- -D warnings
```

### Full Check

```bash
cargo fmt --all && cargo clippy --all && cargo test --all
```

## Documentation

### Generate Docs

```bash
cargo doc --no-deps --open
```

### Doc Comments

```rust
/// Brief description.
///
/// Longer description with details.
///
/// # Arguments
/// * `param1` - Description
///
/// # Returns
/// Description of return value
///
/// # Errors
/// Possible errors
///
/// # Examples
/// ```
/// let result = my_function();
/// ```
pub fn my_function() -> Result<T> {
    // ...
}
```

## Common Tasks

### Add New Module

1. Create file under appropriate crate
2. Add `pub mod name;` to parent mod.rs
3. Write tests in same file or tests/ directory
4. Add documentation

### Add New Optimizer

1. Create `optim/my_optimizer.rs`
2. Implement optimizer trait
3. Add tests
4. Update `optim/mod.rs`

### Add New Neural Network Layer

1. Create `nn/my_layer.rs`
2. Implement layer struct and forward pass
3. Implement autograd support
4. Add tests in `nn/` module
5. Update `nn/mod.rs`

### Add CLI Command

1. Create `commands/my_command.rs`
2. Implement command handler
3. Add to `commands/mod.rs`
4. Update main.rs argument parsing

### Add Python Binding

1. Create wrapper in `onebitllm-python/src/`
2. Use PyO3 macros for Python exposure
3. Add docstrings
4. Test with Python

## Performance Profiling

### Flame Graph (Linux)

```bash
# Install flamegraph
cargo install flamegraph

# Profile
cargo flamegraph -p onebitllm-core --bench inference_bench

# View
open flamegraph.svg
```

### Perf (Linux)

```bash
cargo build --release -p onebitllm-core
perf record -F 99 -p <pid> sleep 30
perf report
```

### Instruments (macOS)

```bash
cargo instruments -t "System Trace" --release
```

## Debugging

### RUST_LOG Levels

```bash
RUST_LOG=trace    # Most detailed
RUST_LOG=debug    # Debug info
RUST_LOG=info     # Informational
RUST_LOG=warn     # Warnings
RUST_LOG=error    # Only errors
```

### Module-Specific Logging

```bash
RUST_LOG=onebitllm_core::quant=debug
RUST_LOG=onebitllm_core::nn=debug
RUST_LOG=onebitllm_cli=info
```

### GDB/LLDB

```bash
# Build with debug symbols
cargo build

# GDB (Linux)
rust-gdb --args target/debug/onebitllm --help

# LLDB (macOS)
rust-lldb -- target/debug/onebitllm --help
```

## Dependency Management

### Update Dependencies

```bash
# Check for updates
cargo update

# Update specific package
cargo update -p ndarray
```

### Add Dependency

```bash
# Add to workspace
cargo add --workspace serde

# Add to specific crate
cargo add -p onebitllm-core serde
```

### Check Licenses

```bash
cargo install cargo-license
cargo license
```

## Continuous Integration

### Local CI Check

```bash
#!/bin/bash
set -e

echo "Formatting..."
cargo fmt --all -- --check

echo "Clippy..."
cargo clippy --all -- -D warnings

echo "Testing..."
cargo test --all

echo "Building..."
cargo build --all --release

echo "All checks passed!"
```

## Release Process

### Version Bump

Update in `Cargo.toml`:

```toml
[workspace.package]
version = "0.2.0"
```

### Changelog

Update CHANGELOG.md with:
- New features
- Bug fixes
- Breaking changes
- Dependencies updates

### Tag Release

```bash
git tag v0.2.0
git push origin v0.2.0
```

## Python Development

### Development Install

```bash
cd crates/onebitllm-python
pip install -e ".[dev]"
```

### Run Python Tests

```bash
cd crates/onebitllm-python
pytest tests/
```

### Build Wheel

```bash
maturin build --release
pip install target/wheels/onebitllm-*.whl
```

## Useful Tools

```bash
# Dependency tree
cargo tree

# Check unused dependencies
cargo udeps

# Outdated dependencies
cargo outdated

# Binary size
cargo bloat --release

# View generated assembly
cargo asm onebitllm_core::quant::bitpack --release
```

## Guidelines

1. **Code Style**: Follow Rust conventions
2. **Testing**: 80%+ coverage for critical code
3. **Documentation**: Document public APIs
4. **Performance**: Benchmark performance-critical code
5. **Errors**: Use custom error types
6. **Dependencies**: Minimize and keep updated
7. **Commits**: Atomic, descriptive commit messages
8. **PRs**: Small, focused changes with tests
