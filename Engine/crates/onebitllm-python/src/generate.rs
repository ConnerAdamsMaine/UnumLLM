use pyo3::prelude::*;

use crate::config::PyGenerateConfig;

/// Text generator for a loaded model.
///
/// This is a placeholder that wraps the generation pipeline.
/// Full functionality requires a loaded model and tokenizer.
#[pyclass]
pub struct PyGenerator {
    config: PyGenerateConfig,
}

#[pymethods]
impl PyGenerator {
    /// Create a new generator with the given configuration.
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyGenerateConfig>) -> Self {
        Self {
            config: config.unwrap_or_else(PyGenerateConfig::default),
        }
    }

    /// Get the generation configuration.
    #[getter]
    fn config(&self) -> PyGenerateConfig {
        self.config.clone()
    }

    /// Generate text from a prompt (placeholder).
    ///
    /// Returns a string. Requires a model and tokenizer to be set up.
    fn generate(&self, prompt: &str) -> PyResult<String> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "Rust generation bindings are not wired to a loaded model/tokenizer yet. \
Received prompt {:?} with max_new_tokens={}.",
            prompt, self.config.max_new_tokens
        )))
    }

    fn __repr__(&self) -> String {
        format!("Generator(max_new_tokens={})", self.config.max_new_tokens)
    }
}
