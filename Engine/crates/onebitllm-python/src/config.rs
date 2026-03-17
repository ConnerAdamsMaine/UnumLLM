use pyo3::prelude::*;

/// Sampling configuration for text generation.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PySamplingConfig {
    /// Temperature for sampling (0.0 = greedy).
    #[pyo3(get, set)]
    pub temperature: f32,
    /// Top-k filtering (None = disabled).
    #[pyo3(get, set)]
    pub top_k: Option<usize>,
    /// Top-p (nucleus) filtering (None = disabled).
    #[pyo3(get, set)]
    pub top_p: Option<f32>,
    /// Repetition penalty (None = disabled).
    #[pyo3(get, set)]
    pub repetition_penalty: Option<f32>,
    /// Random seed for reproducible results.
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl PySamplingConfig {
    #[new]
    #[pyo3(signature = (temperature=0.7, top_k=None, top_p=None, repetition_penalty=None, seed=None))]
    fn new(
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        repetition_penalty: Option<f32>,
        seed: Option<u64>,
    ) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SamplingConfig(temperature={}, top_k={:?}, top_p={:?}, repetition_penalty={:?}, seed={:?})",
            self.temperature, self.top_k, self.top_p, self.repetition_penalty, self.seed
        )
    }
}

impl PySamplingConfig {
    pub fn to_core(&self) -> onebitllm_core::infer::SamplingConfig {
        onebitllm_core::infer::SamplingConfig {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
        }
    }
}

/// Generation configuration.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyGenerateConfig {
    /// Maximum number of new tokens to generate.
    #[pyo3(get, set)]
    pub max_new_tokens: usize,
    /// Sampling configuration.
    #[pyo3(get, set)]
    pub sampling: PySamplingConfig,
    /// Token IDs that stop generation.
    #[pyo3(get, set)]
    pub stop_tokens: Vec<u32>,
}

#[pymethods]
impl PyGenerateConfig {
    #[new]
    #[pyo3(signature = (max_new_tokens=256, sampling=None, stop_tokens=None))]
    fn new(
        max_new_tokens: usize,
        sampling: Option<PySamplingConfig>,
        stop_tokens: Option<Vec<u32>>,
    ) -> Self {
        Self {
            max_new_tokens,
            sampling: sampling.unwrap_or_else(|| PySamplingConfig::new(0.7, None, None, None, None)),
            stop_tokens: stop_tokens.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GenerateConfig(max_new_tokens={}, stop_tokens={:?})",
            self.max_new_tokens, self.stop_tokens
        )
    }
}

impl Default for PyGenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sampling: PySamplingConfig {
                temperature: 0.7,
                top_k: None,
                top_p: None,
                repetition_penalty: None,
                seed: None,
            },
            stop_tokens: Vec::new(),
        }
    }
}

impl PyGenerateConfig {
    pub fn to_core(&self) -> onebitllm_core::infer::GenerateConfig {
        onebitllm_core::infer::GenerateConfig {
            max_new_tokens: self.max_new_tokens,
            sampling: self.sampling.to_core(),
            stop_tokens: self.stop_tokens.clone(),
        }
    }
}

/// Model architecture configuration.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyModelConfig {
    #[pyo3(get, set)]
    pub architecture: String,
    #[pyo3(get, set)]
    pub hidden_size: usize,
    #[pyo3(get, set)]
    pub num_layers: usize,
    #[pyo3(get, set)]
    pub num_attention_heads: usize,
    #[pyo3(get, set)]
    pub num_kv_heads: usize,
    #[pyo3(get, set)]
    pub intermediate_size: usize,
    #[pyo3(get, set)]
    pub vocab_size: usize,
    #[pyo3(get, set)]
    pub max_seq_len: usize,
    #[pyo3(get, set)]
    pub activation: String,
}

#[pymethods]
impl PyModelConfig {
    #[new]
    #[pyo3(signature = (
        architecture="bitnet-b1.58".to_string(),
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        num_kv_heads=12,
        intermediate_size=2048,
        vocab_size=32000,
        max_seq_len=2048,
        activation="silu".to_string(),
    ))]
    fn new(
        architecture: String,
        hidden_size: usize,
        num_layers: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        intermediate_size: usize,
        vocab_size: usize,
        max_seq_len: usize,
        activation: String,
    ) -> Self {
        Self {
            architecture,
            hidden_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            intermediate_size,
            vocab_size,
            max_seq_len,
            activation,
        }
    }

    /// Load config from a JSON file.
    #[staticmethod]
    fn from_json(path: &str) -> PyResult<Self> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let config = onebitllm_core::io::ModelConfig::from_json_str(&data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self::from_core(&config))
    }

    /// Save config to a JSON file.
    fn save_json(&self, path: &str) -> PyResult<()> {
        let core_config = self.to_core();
        let file = std::fs::File::create(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        core_config.save_json(file)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelConfig(arch='{}', hidden={}, layers={}, heads={}, vocab={})",
            self.architecture, self.hidden_size, self.num_layers,
            self.num_attention_heads, self.vocab_size
        )
    }
}

impl PyModelConfig {
    pub fn to_core(&self) -> onebitllm_core::io::ModelConfig {
        onebitllm_core::io::ModelConfig {
            architecture: self.architecture.clone(),
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_seq_len,
            activation: self.activation.clone(),
            ..Default::default()
        }
    }

    pub fn from_core(config: &onebitllm_core::io::ModelConfig) -> Self {
        Self {
            architecture: config.architecture.clone(),
            hidden_size: config.hidden_size,
            num_layers: config.num_layers,
            num_attention_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            intermediate_size: config.intermediate_size,
            vocab_size: config.vocab_size,
            max_seq_len: config.max_seq_len,
            activation: config.activation.clone(),
        }
    }
}
