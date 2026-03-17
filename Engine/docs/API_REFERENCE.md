# API Reference

## Core Library (`onebitllm-core`)

### Tensor Module

#### `PackedTensor`

```rust
pub struct PackedTensor {
    data: Vec<u8>,
    shape: Shape,
}
```

**Methods**:
- `new(shape: Shape) -> Result<Self>` - Create new tensor
- `zeros(shape: Shape) -> Result<Self>` - Create zero-filled tensor
- `from_values(values: &[f32], shape: Shape) -> Result<Self>` - From float values
- `shape(&self) -> &Shape` - Get shape
- `get(&self, indices: &[usize]) -> Result<f32>` - Get element
- `set(&mut self, indices: &[usize], value: f32) -> Result<()>` - Set element
- `reshape(&self, new_shape: Shape) -> Result<Self>` - Reshape tensor

#### `Shape`

```rust
pub struct Shape {
    dims: Vec<usize>,
    strides: Vec<usize>,
}
```

**Methods**:
- `new(dims: Vec<usize>) -> Result<Self>` - Create shape
- `dims(&self) -> &[usize]` - Get dimensions
- `numel(&self) -> usize` - Total number of elements
- `is_contiguous(&self) -> bool` - Check if contiguous
- `broadcast_to(&self, target: &Shape) -> Result<Shape>` - Broadcast shape

#### Tensor Operations

```rust
// Element-wise operations
pub fn add(a: &PackedTensor, b: &PackedTensor) -> Result<PackedTensor>;
pub fn sub(a: &PackedTensor, b: &PackedTensor) -> Result<PackedTensor>;
pub fn mul(a: &PackedTensor, b: &PackedTensor) -> Result<PackedTensor>;
pub fn div(a: &PackedTensor, b: &PackedTensor) -> Result<PackedTensor>;

// Reduction operations
pub fn sum(tensor: &PackedTensor, axis: Option<usize>) -> Result<PackedTensor>;
pub fn mean(tensor: &PackedTensor, axis: Option<usize>) -> Result<PackedTensor>;
pub fn max(tensor: &PackedTensor, axis: Option<usize>) -> Result<PackedTensor>;
pub fn min(tensor: &PackedTensor, axis: Option<usize>) -> Result<PackedTensor>;

// Shape operations
pub fn transpose(tensor: &PackedTensor, axes: &[usize]) -> Result<PackedTensor>;
pub fn reshape(tensor: &PackedTensor, shape: Shape) -> Result<PackedTensor>;
pub fn broadcast(tensor: &PackedTensor, shape: Shape) -> Result<PackedTensor>;
pub fn matmul(a: &PackedTensor, b: &PackedTensor) -> Result<PackedTensor>;
```

### Quantization Module

#### `QuantConfig`

```rust
pub struct QuantConfig {
    pub granularity: QuantGranularity,
    pub scale_dtype: String,
    pub learnable_scales: bool,
    pub ste_temperature: f32,
}
```

**Granularities**:
- `PerTensor` - One scale for entire tensor
- `PerChannel` - One scale per channel
- `PerGroup(usize)` - One scale per group

**Methods**:
- `new(granularity: QuantGranularity) -> Self`
- `with_learnable_scales(mut self, learnable: bool) -> Self`
- `with_ste_temperature(mut self, temp: f32) -> Self`

#### `QuantParams`

```rust
pub struct QuantParams {
    scales: PackedTensor,
    zero_points: PackedTensor,
}
```

**Methods**:
- `new(config: &QuantConfig, shape: Shape) -> Result<Self>`
- `quantize(&self, weights: &PackedTensor) -> Result<PackedTensor>`
- `dequantize(&self, quantized: &PackedTensor) -> Result<PackedTensor>`
- `update_scales(&mut self, weights: &PackedTensor) -> Result<()>`

#### `TernaryWeight`

```rust
#[repr(i8)]
pub enum TernaryWeight {
    Negative = -1,
    Zero = 0,
    Positive = 1,
}
```

**Methods**:
- `from_float(value: f32) -> Self` - Convert from float
- `to_float(&self) -> f32` - Convert to float

#### `PackedTernary`

```rust
pub struct PackedTernary {
    data: Vec<u8>,
}
```

**Methods**:
- `pack(weights: &[TernaryWeight]) -> Self`
- `unpack(&self, count: usize) -> Vec<TernaryWeight>`

### Neural Network Module

#### `Linear`

```rust
pub struct QuantizedLinear {
    weight: PackedTensor,
    bias: Option<PackedTensor>,
    quant_params: QuantParams,
}
```

**Methods**:
- `new(in_features: usize, out_features: usize, config: &QuantConfig) -> Result<Self>`
- `forward(&self, input: &PackedTensor) -> Result<PackedTensor>`
- `backward(&mut self, grad_output: &PackedTensor) -> Result<PackedTensor>`

#### `Attention`

```rust
pub struct Attention {
    config: AttentionConfig,
    q_proj: QuantizedLinear,
    k_proj: QuantizedLinear,
    v_proj: QuantizedLinear,
    out_proj: QuantizedLinear,
    pos_encoding: PositionalEncoding,
}
```

**Config**:
```rust
pub struct AttentionConfig {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub dropout: f32,
    pub attention_dropout: f32,
}
```

**Methods**:
- `new(config: &AttentionConfig) -> Result<Self>`
- `forward(&self, input: &PackedTensor) -> Result<PackedTensor>`
- `forward_with_cache(&mut self, input: &PackedTensor, cache: &mut KVCache) -> Result<PackedTensor>`

#### `MlpBlock`

```rust
pub struct MlpBlock {
    up_proj: QuantizedLinear,
    down_proj: QuantizedLinear,
    activation: ActivationFn,
    dropout: f32,
}
```

**Methods**:
- `new(hidden_dim: usize, intermediate_dim: usize, activation: ActivationFn) -> Result<Self>`
- `forward(&self, input: &PackedTensor) -> Result<PackedTensor>`

#### `Embedding`

```rust
pub struct Embedding {
    weight: PackedTensor,
}
```

**Methods**:
- `new(vocab_size: usize, embedding_dim: usize) -> Result<Self>`
- `forward(&self, input: &PackedTensor) -> Result<PackedTensor>`

#### Normalization

```rust
pub struct RmsNorm {
    weight: PackedTensor,
    eps: f32,
}

pub struct LayerNorm {
    weight: PackedTensor,
    bias: PackedTensor,
    eps: f32,
}
```

**Methods**:
- `new(dim: usize) -> Result<Self>`
- `forward(&self, input: &PackedTensor) -> Result<PackedTensor>`

#### Positional Encodings

```rust
pub struct RotaryEmbedding {
    inv_freq: Vec<f32>,
}

pub struct AlibiEmbedding {
    num_heads: usize,
}

pub struct LearnedPositionalEmbedding {
    embeddings: PackedTensor,
}
```

**Methods**:
- `new(hidden_dim: usize) -> Result<Self>`
- `encode(&self, seq_len: usize) -> Result<PackedTensor>`

#### Activation Functions

```rust
pub enum ActivationFn {
    ReLU,
    GELU,
    SiLU,
    Swish,
    Tanh,
}
```

**Methods**:
- `forward(&self, input: &PackedTensor) -> Result<PackedTensor>`

### Training Module

#### `Trainer`

```rust
pub struct Trainer {
    model: Model,
    optimizer: Box<dyn Optimizer>,
    config: TrainConfig,
}
```

**Methods**:
- `new(model: Model, optimizer: Box<dyn Optimizer>, config: TrainConfig) -> Result<Self>`
- `train_batch(&mut self, input: &PackedTensor, target: &PackedTensor) -> Result<f32>`
- `step(&mut self) -> Result<()>`
- `save_checkpoint(&self, path: &str) -> Result<()>`

#### `TrainConfig`

```rust
pub struct TrainConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub warmup_steps: usize,
    pub grad_clip: Option<f32>,
}
```

### Optimization Module

#### `Optimizer` Trait

```rust
pub trait Optimizer {
    fn step(&mut self, grads: &[PackedTensor], params: &mut [PackedTensor]) -> Result<()>;
    fn learning_rate(&self) -> f32;
}
```

#### `Sgd`

```rust
pub struct Sgd {
    lr: f32,
    momentum: Option<f32>,
}
```

**Methods**:
- `new(lr: f32) -> Self`
- `with_momentum(mut self, momentum: f32) -> Self`

#### `Adam`

```rust
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
}
```

**Methods**:
- `new(lr: f32) -> Self`
- `with_betas(mut self, beta1: f32, beta2: f32) -> Self`

#### `AdamW`

```rust
pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}
```

**Methods**:
- `new(lr: f32, weight_decay: f32) -> Self`

### Inference Module

#### `Inferencer`

```rust
pub struct Inferencer {
    model: Model,
    config: InferConfig,
}
```

**Methods**:
- `new(model: Model, config: InferConfig) -> Result<Self>`
- `generate(&mut self, prompt_ids: &[usize], max_tokens: usize) -> Result<Vec<usize>>`
- `generate_batch(&mut self, prompts: &[&[usize]], max_tokens: usize) -> Result<Vec<Vec<usize>>>`

#### `KVCache`

```rust
pub struct KVCache {
    keys: PackedTensor,
    values: PackedTensor,
}
```

**Methods**:
- `new(max_seq_len: usize, hidden_dim: usize) -> Result<Self>`
- `update(&mut self, k: &PackedTensor, v: &PackedTensor) -> Result<()>`
- `clear(&mut self)`

### I/O Module

#### Save/Load

```rust
pub fn save_checkpoint(model: &Model, path: &str) -> Result<()>;
pub fn load_checkpoint(path: &str) -> Result<Model>;

pub fn save_safetensors(tensors: &HashMap<String, PackedTensor>, path: &str) -> Result<()>;
pub fn load_safetensors(path: &str) -> Result<HashMap<String, PackedTensor>>;
```

#### Config

```rust
pub fn load_yaml_config(path: &str) -> Result<ModelConfig>;
pub fn save_yaml_config(config: &ModelConfig, path: &str) -> Result<()>;
```

### Error Handling

```rust
pub enum OneBitError {
    ShapeError(String),
    QuantizationError(String),
    ComputeError(String),
    SerializationError(String),
    IoError(String),
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, OneBitError>;
```

## CLI (`onebitllm-cli`)

### Commands

Current status: the Rust CLI validates arguments and reports implementation status, but the end-to-end train, quantize, and generate pipelines are not wired yet.

#### train

```bash
onebitllm train [OPTIONS]

OPTIONS:
  --config <FILE>
  --data <FILE>
  --output <DIR>
  --epochs <N>
  --batch-size <N>
  --lr <LR>
  --warmup-steps <N>
  --save-every <N>
  --seed <SEED>
  --resume <FILE>
```

#### quantize

```bash
onebitllm quantize [OPTIONS]

OPTIONS:
  --input <FILE>
  --output <FILE>
  --granularity <GRAN>
  --group-size <N>
```

#### generate

```bash
onebitllm generate [OPTIONS]

OPTIONS:
  --model <FILE>
  --prompt <TEXT>
  --max-tokens <N>
  --temperature <T>
  --top-k <K>
  --top-p <P>
  --repetition-penalty <P>
  --stream
```

## Python API (`onebitllm-python`)

Current note: these bindings expose the Rust surface directly. Model loading and
configuration/tokenizer helpers exist, while generation and live-model save
paths still fail explicitly until the runtime wiring is finished.

### Classes

#### `PyModelConfig`

```python
class PyModelConfig:
    def __init__(
        self,
        architecture: str = "bitnet-b1.58",
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        num_kv_heads: int = 12,
        intermediate_size: int = 2048,
        vocab_size: int = 32000,
        max_seq_len: int = 2048,
        activation: str = "silu",
    ) -> None: ...

    @staticmethod
    def from_json(path: str) -> "PyModelConfig": ...

    def save_json(self, path: str) -> None: ...
```

#### `PyModel`

```python
class PyModel:
    def __init__(self, config: PyModelConfig) -> None: ...

    @staticmethod
    def load(path: str) -> "PyModel": ...

    def save(self, path: str) -> None: ...
    @property
    def config(self) -> PyModelConfig: ...
    @property
    def num_parameters(self) -> int: ...
```

#### `PyGenerateConfig`

```python
class PyGenerateConfig:
    def __init__(
        self,
        max_new_tokens: int = 256,
        sampling: Optional[PySamplingConfig] = None,
        stop_tokens: Optional[List[int]] = None,
    ) -> None: ...
```

#### `PyGenerator`

```python
class PyGenerator:
    def __init__(self, config: Optional[PyGenerateConfig] = None) -> None: ...

    @property
    def config(self) -> PyGenerateConfig: ...

    def generate(self, prompt: str) -> str: ...
```

#### `PyTokenizer`

```python
class PyTokenizer:
    def __init__(self, vocab: Dict[str, int], merges: List[Tuple[str, str]]) -> None: ...

    def encode(self, text: str) -> PyEncoding: ...
    def decode(self, ids: List[int]) -> str: ...
    def vocab_size(self) -> int: ...
```

## Types and Enums

### Quantization

```rust
pub enum QuantGranularity {
    PerTensor,
    PerChannel,
    PerGroup(usize),
}
```

### Activation Functions

```rust
pub enum ActivationFn {
    ReLU,
    GELU,
    SiLU,
    Swish,
    Tanh,
}
```

### Positional Encodings

```rust
pub enum PositionalEncoding {
    Rotary,
    Alibi,
    Learned,
}
```
