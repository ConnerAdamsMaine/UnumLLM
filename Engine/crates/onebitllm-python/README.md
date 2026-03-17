# onebitllm-python

Python bindings for 1-bit quantized LLM inference engine.

## Overview

`onebitllm-python` provides high-level Python APIs for loading, configuring, and running inference with 1-bit quantized language models. It wraps `onebitllm-core` using PyO3, enabling seamless integration with Python workflows.

## Installation

### From Source

```bash
pip install maturin
cd crates/onebitllm-python
maturin develop --release

# Or directly from workspace root:
maturin develop --release -m crates/onebitllm-python/Cargo.toml
```

### Build as Wheel

```bash
maturin build --release -m crates/onebitllm-python/Cargo.toml
pip install target/wheels/onebitllm-*.whl
```

## Quick Start

```python
import onebitllm

# Create model configuration
config = onebitllm.ModelConfig(
    vocab_size=10000,
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    intermediate_dim=3072,
    max_seq_len=2048,
)

# Initialize model
model = onebitllm.OneBitModel(config)

# Load pretrained weights (optional)
model.load("path/to/model.pt")

# Generate text
prompt = "The future of AI is"
generated = model.generate(prompt, max_tokens=100)
print(generated)
```

## API Reference

### ModelConfig

Configuration for model architecture and hyperparameters.

```python
config = onebitllm.ModelConfig(
    vocab_size: int,           # Vocabulary size
    hidden_dim: int,           # Hidden dimension
    num_layers: int,           # Number of transformer layers
    num_heads: int,            # Number of attention heads
    intermediate_dim: int,     # MLP hidden dimension
    max_seq_len: int,          # Maximum sequence length
    positional_encoding: str = "rotary",  # "rotary", "alibi", "learned"
    activation: str = "gelu",  # "gelu", "relu", "silu"
    dropout: float = 0.1,      # Dropout rate
    attention_dropout: float = 0.1,
    layer_norm_type: str = "rms",  # "rms" or "layer"
)
```

**Attributes**:
- `vocab_size`: Number of tokens in vocabulary
- `hidden_dim`: Size of hidden representations
- `num_layers`: Depth of transformer stack
- `num_heads`: Number of attention heads (must divide hidden_dim)
- `intermediate_dim`: Width of MLP intermediate layer
- `max_seq_len`: Maximum input sequence length
- `positional_encoding`: Type of positional encoding
- `activation`: Activation function in MLP
- `dropout`: General dropout probability
- `attention_dropout`: Attention-specific dropout
- `layer_norm_type`: Layer normalization variant

### OneBitModel

Main model class for inference.

```python
model = onebitllm.OneBitModel(config)
```

**Methods**:

#### `generate(prompt: str, max_tokens: int, **kwargs) -> str`

Generate text from a prompt.

```python
output = model.generate(
    prompt="Hello, world",
    max_tokens=100,
    temperature=0.7,           # Optional: sampling temperature
    top_p=0.9,                 # Optional: nucleus sampling
    top_k=50,                  # Optional: top-k sampling
)
```

**Parameters**:
- `prompt`: Input text
- `max_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature (default: 1.0)
  - < 1.0: More deterministic
  - = 1.0: Standard sampling
  - > 1.0: More random
- `top_p`: Nucleus sampling cutoff (default: 1.0, disabled)
- `top_k`: Keep top-k tokens by probability (default: None)
- `seed`: Random seed for reproducibility (optional)

**Returns**: Generated text as string

#### `encode(text: str) -> List[int]`

Tokenize text to token IDs.

```python
tokens = model.encode("Hello, world")
# [1234, 5678, 9012, ...]
```

**Parameters**:
- `text`: Input text

**Returns**: List of token IDs

#### `decode(tokens: List[int]) -> str`

Convert token IDs to text.

```python
text = model.decode([1234, 5678, 9012])
# "Hello, world"
```

**Parameters**:
- `tokens`: List of token IDs

**Returns**: Decoded text

#### `save(path: str)`

Save model checkpoint to disk.

```python
model.save("my_model.pt")
```

**Parameters**:
- `path`: Output file path

#### `load(path: str)`

Load model weights from checkpoint.

```python
model.load("my_model.pt")
```

**Parameters**:
- `path`: Checkpoint file path

#### `config() -> ModelConfig`

Get current model configuration.

```python
cfg = model.config()
print(f"Hidden dim: {cfg.hidden_dim}")
```

**Returns**: ModelConfig object

### Tokenizer

Optional standalone tokenizer interface.

```python
tokenizer = onebitllm.Tokenizer(vocab_path="vocab.txt")
tokens = tokenizer.encode("Hello, world")
text = tokenizer.decode([1234, 5678])
```

## Examples

### Basic Text Generation

```python
import onebitllm

# Create and configure model
config = onebitllm.ModelConfig(
    vocab_size=50000,
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    intermediate_dim=3072,
    max_seq_len=2048,
)

model = onebitllm.OneBitModel(config)
model.load("quantized_model.pt")

# Generate diverse outputs with different temperatures
prompts = [
    "The capital of France is",
    "Machine learning is",
    "To solve this problem, we",
]

for prompt in prompts:
    print(f"Prompt: {prompt}")
    print(f"Output: {model.generate(prompt, max_tokens=30)}")
    print()
```

### Batch Tokenization

```python
texts = [
    "First example text",
    "Second example text",
    "Third example text",
]

model = onebitllm.OneBitModel(config)

# Tokenize all texts
all_tokens = [model.encode(text) for text in texts]

# Process batches in your own code
batch_size = 2
for i in range(0, len(all_tokens), batch_size):
    batch = all_tokens[i:i+batch_size]
    print(f"Batch {i//batch_size}: {batch}")
```

### Interactive Chat Loop

```python
import onebitllm

config = onebitllm.ModelConfig(...)
model = onebitllm.OneBitModel(config)
model.load("chat_model.pt")

print("Enter 'quit' to exit")
while True:
    prompt = input("You: ").strip()
    if prompt.lower() == "quit":
        break
    
    response = model.generate(
        prompt,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9
    )
    print(f"Bot: {response}")
```

### Model Quantization Workflow

```python
import onebitllm

# Load full-precision model
config = onebitllm.ModelConfig(...)
model = onebitllm.OneBitModel(config)
model.load("full_precision_model.pt")

# Quantize to 1-bit
quantization_config = {
    "granularity": "channel",
    "group_size": 128,
    "learnable_scales": True,
}
model.quantize(quantization_config)

# Save quantized version
model.save("quantized_model.pt")

# Inference is now faster
output = model.generate("Hello", max_tokens=50)
```

## Performance

### Memory Usage

1-bit quantization achieves 32× weight reduction:

```
Full precision (FP32):  4 bytes per weight
1-bit quantized:        0.125 bytes per weight (1 byte stores 8 values)
→ 32× smaller models
```

### Inference Speed

Typical inference speeds (1-bit quantized model):

| Model Size | Batch Size | Tokens/Second |
|------------|-----------|---------------|
| 110M       | 1         | ~50           |
| 110M       | 8         | ~300          |
| 1B         | 1         | ~10           |
| 1B         | 8         | ~50           |

*Approximate values; vary by hardware and configuration*

## Troubleshooting

### ImportError: Cannot import onebitllm

Ensure proper installation:
```bash
pip install --upgrade maturin
maturin develop --release -m crates/onebitllm-python/Cargo.toml
```

### Model file not found

Check file path is correct and file exists:
```python
import os
if os.path.exists("my_model.pt"):
    model.load("my_model.pt")
else:
    print("Model file not found")
```

### Generation is slow

- Reduce `max_tokens`
- Use smaller model
- Check hardware utilization
- Try batch inference if processing multiple sequences

### Out of memory errors

- Reduce model size (smaller `hidden_dim`, fewer `num_layers`)
- Reduce `max_seq_len`
- Lower batch size (if using batching)

## Integration with HuggingFace

```python
import onebitllm
from transformers import AutoTokenizer

# Use HuggingFace tokenizer with OneBitLLM model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = onebitllm.OneBitModel(config)
model.load("quantized_model.pt")

# Encode with HF tokenizer
tokens = tokenizer.encode("Hello, world")

# Or use integrated tokenizer
tokens = model.encode("Hello, world")
```

## Advanced Usage

### Custom Configuration from Dict

```python
import onebitllm

config_dict = {
    "vocab_size": 50000,
    "hidden_dim": 768,
    "num_layers": 12,
    "num_heads": 12,
    "intermediate_dim": 3072,
    "max_seq_len": 2048,
}

config = onebitllm.ModelConfig(**config_dict)
model = onebitllm.OneBitModel(config)
```

### Streaming Generation

```python
# Not currently supported - full generation returns complete output
# For streaming, collect partial results by reducing max_tokens
# and calling generate multiple times
```

## Contributing

Obtain prior written permission from the copyright holder before preparing or
submitting any contribution.

When extending the Python API:

1. Modify `src/*.rs` files
2. Update docstrings with examples
3. Test with: `maturin develop && python -m pytest`
4. Rebuild before testing changes

## Dependencies

- **PyO3 0.28+**: Python bindings
- **onebitllm-core**: Core Rust library
- **Python 3.8+**: Runtime

## License

See the custom [OneBitLLM Research Review Only License 1.0](../../LICENSE).
