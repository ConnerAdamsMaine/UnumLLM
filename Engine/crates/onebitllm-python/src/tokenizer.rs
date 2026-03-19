use pyo3::prelude::*;

use onebitllm_core::tokenizer::bpe::SimpleBpe;
use onebitllm_core::tokenizer::{Encoding, Tokenizer};

/// A Python-accessible tokenizer.
///
/// Currently wraps the built-in SimpleBpe tokenizer.
/// Extensible to support HuggingFace tokenizers via the `tokenizers-hf` feature.
#[pyclass]
pub struct PyTokenizer {
    inner: Box<dyn Tokenizer + Send>,
}

#[pymethods]
impl PyTokenizer {
    /// Create a simple BPE tokenizer from vocabulary and merge rules.
    ///
    /// Args:
    ///     vocab: dict mapping token strings to integer IDs
    ///     merges: list of (str, str) merge pairs in priority order
    #[new]
    fn new(
        vocab: std::collections::HashMap<String, u32>,
        merges: Vec<(String, String)>,
    ) -> PyResult<Self> {
        let bpe = SimpleBpe::from_data(vocab, merges);
        Ok(Self {
            inner: Box::new(bpe),
        })
    }

    /// Encode text into token IDs.
    ///
    /// Returns a dict with 'ids', 'tokens', and 'attention_mask'.
    fn encode(&self, text: &str) -> PyResult<PyEncoding> {
        let encoding = self
            .inner
            .encode(text)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyEncoding::from_core(encoding))
    }

    /// Decode token IDs back to text.
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner
            .decode(&ids)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Return the vocabulary size.
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.inner.vocab_size())
    }
}

/// Encoding result from tokenization.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyEncoding {
    /// Token IDs.
    #[pyo3(get)]
    pub ids: Vec<u32>,
    /// Token strings.
    #[pyo3(get)]
    pub tokens: Vec<String>,
    /// Attention mask (1 for real tokens, 0 for padding).
    #[pyo3(get)]
    pub attention_mask: Vec<u32>,
}

#[pymethods]
impl PyEncoding {
    fn __repr__(&self) -> String {
        format!("Encoding(ids={:?}, len={})", self.ids, self.ids.len())
    }

    fn __len__(&self) -> usize {
        self.ids.len()
    }
}

impl PyEncoding {
    fn from_core(enc: Encoding) -> Self {
        Self {
            ids: enc.ids,
            tokens: enc.tokens,
            attention_mask: enc.attention_mask,
        }
    }
}
