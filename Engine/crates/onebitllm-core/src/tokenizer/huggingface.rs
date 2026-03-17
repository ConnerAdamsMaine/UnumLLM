use std::path::Path;

use tokenizers::Tokenizer as HfTokenizerInner;

use crate::Result;
use crate::error::OneBitError;
use super::traits::{Encoding, Tokenizer};

/// HuggingFace Tokenizers adapter.
///
/// Wraps the `tokenizers` crate to provide our `Tokenizer` trait interface.
/// Supports loading from pretrained models or local files.
pub struct HuggingFaceTokenizer {
    inner: HfTokenizerInner,
}

impl HuggingFaceTokenizer {
    /// Load from a pretrained model on the HuggingFace Hub.
    ///
    /// Requires network access. Example: `"gpt2"`, `"bert-base-uncased"`.
    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let inner = HfTokenizerInner::from_pretrained(model_name, None)
            .map_err(|e| OneBitError::Tokenizer(format!("Failed to load pretrained tokenizer '{model_name}': {e}")))?;
        Ok(Self { inner })
    }

    /// Load from a local `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = HfTokenizerInner::from_file(path)
            .map_err(|e| OneBitError::Tokenizer(format!("Failed to load tokenizer from {}: {e}", path.display())))?;
        Ok(Self { inner })
    }
}

impl Tokenizer for HuggingFaceTokenizer {
    fn encode(&self, text: &str) -> Result<Encoding> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| OneBitError::Tokenizer(e.to_string()))?;

        Ok(Encoding {
            ids: encoding.get_ids().to_vec(),
            tokens: encoding.get_tokens().to_vec(),
            attention_mask: encoding.get_attention_mask().to_vec(),
        })
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| OneBitError::Tokenizer(e.to_string()))
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

// Tests require network access, marked #[ignore] for CI.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires network access to download tokenizer"]
    fn test_hf_gpt2_roundtrip() {
        let tok = HuggingFaceTokenizer::from_pretrained("gpt2").unwrap();
        let text = "Hello, world!";
        let enc = tok.encode(text).unwrap();
        assert!(!enc.ids.is_empty());

        let decoded = tok.decode(&enc.ids).unwrap();
        assert_eq!(decoded.trim(), text);
    }

    #[test]
    #[ignore = "requires network access to download tokenizer"]
    fn test_hf_vocab_size() {
        let tok = HuggingFaceTokenizer::from_pretrained("gpt2").unwrap();
        assert!(tok.vocab_size() > 50000);
    }
}
