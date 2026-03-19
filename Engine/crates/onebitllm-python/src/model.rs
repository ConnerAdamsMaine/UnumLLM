use pyo3::prelude::*;

use crate::config::PyModelConfig;

/// A 1-bit quantized model.
///
/// This is a placeholder that will be connected to the actual model
/// implementation once a full model architecture (transformer block stack)
/// is assembled from the nn module components.
#[pyclass]
pub struct PyModel {
    config: PyModelConfig,
    num_parameters: usize,
}

#[pymethods]
impl PyModel {
    /// Create a new model from a configuration.
    #[new]
    fn new(config: PyModelConfig) -> Self {
        // Estimate parameter count from config
        let h = config.hidden_size;
        let l = config.num_layers;
        let ff = config.intermediate_size;
        let v = config.vocab_size;

        // Rough estimate: embedding + per-layer (attn + ffn + norms)
        let embedding_params = v * h;
        let attn_params = 4 * h * h; // Q, K, V, O projections
        let ffn_params = 3 * h * ff; // up, gate, down
        let norm_params = 2 * h; // 2 norms per layer
        let per_layer = attn_params + ffn_params + norm_params;
        let total = embedding_params + l * per_layer + h; // +final norm

        Self {
            config,
            num_parameters: total,
        }
    }

    /// Load a model from an OBM file.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let obm = onebitllm_core::io::ObmFile::load(file)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let config = PyModelConfig::from_core(&obm.header.config);
        let num_params: usize = obm
            .tensors
            .iter()
            .map(|t| t.shape.iter().product::<usize>())
            .sum();

        Ok(Self {
            config,
            num_parameters: num_params,
        })
    }

    /// Save the model to an OBM file.
    fn save(&self, path: &str) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "Saving a live Rust model is not implemented yet. Refusing to write placeholder data to {:?}.",
            path
        )))
    }

    /// Get the model configuration.
    #[getter]
    fn config(&self) -> PyModelConfig {
        self.config.clone()
    }

    /// Total number of parameters.
    #[getter]
    fn num_parameters(&self) -> usize {
        self.num_parameters
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(arch='{}', params={:.1}M)",
            self.config.architecture,
            self.num_parameters as f64 / 1e6
        )
    }
}
