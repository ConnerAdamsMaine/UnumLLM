use std::path::PathBuf;

/// Result of encoding text into tokens.
#[derive(Debug, Clone)]
pub struct Encoding {
    /// Token IDs.
    pub ids: Vec<u32>,
    /// Token strings (for debugging/display).
    pub tokens: Vec<String>,
    /// Attention mask (1 for real tokens, 0 for padding).
    pub attention_mask: Vec<u32>,
}

/// Configuration for loading a tokenizer.
#[derive(Debug, Clone)]
pub enum TokenizerConfig {
    /// Load from a HuggingFace model name (downloads from hub).
    HuggingFace { model_name: String },
    /// Load from a local tokenizer.json file.
    File { path: PathBuf },
    /// Custom BPE with vocab and merges files.
    Bpe {
        vocab_path: PathBuf,
        merges_path: PathBuf,
    },
}

/// The pluggable tokenizer trait.
///
/// All tokenizer implementations must be thread-safe (`Send + Sync`).
pub trait Tokenizer: Send + Sync {
    /// Encode text to token IDs.
    fn encode(&self, text: &str) -> crate::Result<Encoding>;

    /// Decode token IDs back to text.
    fn decode(&self, ids: &[u32]) -> crate::Result<String>;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Encode with padding/truncation to a fixed length.
    fn encode_padded(&self, text: &str, max_len: usize, pad_id: u32) -> crate::Result<Encoding> {
        let mut enc = self.encode(text)?;

        if enc.ids.len() > max_len {
            enc.ids.truncate(max_len);
            enc.tokens.truncate(max_len);
            enc.attention_mask.truncate(max_len);
        } else {
            while enc.ids.len() < max_len {
                enc.ids.push(pad_id);
                enc.tokens.push("<pad>".to_string());
                enc.attention_mask.push(0);
            }
        }

        Ok(enc)
    }

    /// Encode a batch of texts.
    fn encode_batch(&self, texts: &[&str]) -> crate::Result<Vec<Encoding>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal test tokenizer for unit testing
    struct CharTokenizer;

    impl Tokenizer for CharTokenizer {
        fn encode(&self, text: &str) -> crate::Result<Encoding> {
            let ids: Vec<u32> = text.chars().map(|c| c as u32).collect();
            let tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
            let attention_mask = vec![1u32; ids.len()];
            Ok(Encoding {
                ids,
                tokens,
                attention_mask,
            })
        }

        fn decode(&self, ids: &[u32]) -> crate::Result<String> {
            Ok(ids.iter().filter_map(|&id| char::from_u32(id)).collect())
        }

        fn vocab_size(&self) -> usize {
            256 // ASCII
        }
    }

    #[test]
    fn test_char_tokenizer_roundtrip() {
        let tok = CharTokenizer;
        let text = "hello";
        let enc = tok.encode(text).unwrap();
        assert_eq!(enc.ids.len(), 5);
        assert_eq!(enc.attention_mask, vec![1, 1, 1, 1, 1]);

        let decoded = tok.decode(&enc.ids).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_padded() {
        let tok = CharTokenizer;

        // Padding
        let enc = tok.encode_padded("hi", 5, 0).unwrap();
        assert_eq!(enc.ids.len(), 5);
        assert_eq!(enc.attention_mask, vec![1, 1, 0, 0, 0]);

        // Truncation
        let enc = tok.encode_padded("hello world", 5, 0).unwrap();
        assert_eq!(enc.ids.len(), 5);
        assert_eq!(enc.attention_mask, vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_encode_batch() {
        let tok = CharTokenizer;
        let texts = vec!["hi", "bye"];
        let batch = tok.encode_batch(&texts).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].ids.len(), 2);
        assert_eq!(batch[1].ids.len(), 3);
    }

    #[test]
    fn test_trait_object() {
        let tok: Box<dyn Tokenizer> = Box::new(CharTokenizer);
        let enc = tok.encode("test").unwrap();
        assert_eq!(enc.ids.len(), 4);
    }
}
