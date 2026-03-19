mod config;
mod generate;
mod model;
mod tokenizer;

use pyo3::prelude::*;

/// OneBitLLM — Python bindings for 1-bit quantized LLM engine.
///
/// Provides high-level Python API for:
/// - Loading and saving 1-bit quantized models
/// - Tokenization
/// - Text generation with configurable sampling
/// - Model configuration
#[pymodule]
fn onebitllm(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<config::PySamplingConfig>()?;
    m.add_class::<config::PyGenerateConfig>()?;
    m.add_class::<config::PyModelConfig>()?;
    m.add_class::<tokenizer::PyTokenizer>()?;
    m.add_class::<model::PyModel>()?;
    m.add_class::<generate::PyGenerator>()?;

    // Module-level functions
    m.add_function(wrap_pyfunction!(version, m)?)?;

    Ok(())
}

/// Return the version string of the onebitllm library.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
