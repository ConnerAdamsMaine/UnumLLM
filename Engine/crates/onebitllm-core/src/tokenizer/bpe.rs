use std::collections::HashMap;
use std::path::Path;

use super::traits::{Encoding, Tokenizer};
use crate::error::OneBitError;
use crate::Result;

/// A minimal BPE (Byte-Pair Encoding) tokenizer.
///
/// This provides basic BPE functionality for users who don't want the
/// HuggingFace `tokenizers` dependency. For production use, prefer the
/// HuggingFace adapter.
pub struct SimpleBpe {
    /// Token string -> ID mapping.
    vocab: HashMap<String, u32>,
    /// ID -> Token string mapping.
    reverse_vocab: HashMap<u32, String>,
    /// Merge rules: pairs of tokens to merge, in priority order.
    merges: Vec<(String, String)>,
}

impl SimpleBpe {
    /// Load from vocab and merges files.
    ///
    /// Vocab file: one token per line, format: `token id`
    /// Merges file: one merge per line, format: `token_a token_b`
    pub fn from_files(vocab_path: &Path, merges_path: &Path) -> Result<Self> {
        let vocab_text = std::fs::read_to_string(vocab_path)?;
        let merges_text = std::fs::read_to_string(merges_path)?;

        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();

        for line in vocab_text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.rsplitn(2, ' ').collect();
            if parts.len() != 2 {
                return Err(OneBitError::Tokenizer(format!(
                    "Invalid vocab line: {line}"
                )));
            }
            let id: u32 = parts[0]
                .parse()
                .map_err(|_| OneBitError::Tokenizer(format!("Invalid ID in vocab: {line}")))?;
            let token = parts[1].to_string();
            reverse_vocab.insert(id, token.clone());
            vocab.insert(token, id);
        }

        let mut merges = Vec::new();
        for line in merges_text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() != 2 {
                return Err(OneBitError::Tokenizer(format!(
                    "Invalid merge line: {line}"
                )));
            }
            merges.push((parts[0].to_string(), parts[1].to_string()));
        }

        Ok(Self {
            vocab,
            reverse_vocab,
            merges,
        })
    }

    /// Create from in-memory vocab and merges (for testing).
    pub fn from_data(vocab: HashMap<String, u32>, merges: Vec<(String, String)>) -> Self {
        let reverse_vocab: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        Self {
            vocab,
            reverse_vocab,
            merges,
        }
    }

    /// Apply BPE merges to a list of tokens.
    fn apply_merges(&self, tokens: &[String]) -> Vec<String> {
        let mut result = tokens.to_vec();

        for (a, b) in &self.merges {
            let mut i = 0;
            while i + 1 < result.len() {
                if &result[i] == a && &result[i + 1] == b {
                    let merged = format!("{a}{b}");
                    result[i] = merged;
                    result.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        result
    }
}

impl Tokenizer for SimpleBpe {
    fn encode(&self, text: &str) -> Result<Encoding> {
        // Start with character-level tokens
        let char_tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        // Apply BPE merges
        let merged = self.apply_merges(&char_tokens);

        // Look up IDs
        let mut ids = Vec::with_capacity(merged.len());
        let mut tokens = Vec::with_capacity(merged.len());

        for token in &merged {
            if let Some(&id) = self.vocab.get(token) {
                ids.push(id);
                tokens.push(token.clone());
            } else {
                // Unknown token fallback: encode as individual bytes
                for ch in token.chars() {
                    let ch_str = ch.to_string();
                    if let Some(&id) = self.vocab.get(&ch_str) {
                        ids.push(id);
                        tokens.push(ch_str);
                    } else {
                        // Use a special UNK id (0)
                        ids.push(0);
                        tokens.push("<unk>".to_string());
                    }
                }
            }
        }

        let attention_mask = vec![1u32; ids.len()];

        Ok(Encoding {
            ids,
            tokens,
            attention_mask,
        })
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut result = String::new();
        for &id in ids {
            if let Some(token) = self.reverse_vocab.get(&id) {
                result.push_str(token);
            } else {
                result.push('\u{FFFD}'); // Unicode replacement character
            }
        }
        Ok(result)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_bpe() -> SimpleBpe {
        let mut vocab = HashMap::new();
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("h".to_string(), 1);
        vocab.insert("e".to_string(), 2);
        vocab.insert("l".to_string(), 3);
        vocab.insert("o".to_string(), 4);
        vocab.insert("he".to_string(), 5);
        vocab.insert("ll".to_string(), 6);
        vocab.insert("llo".to_string(), 7);
        vocab.insert("hello".to_string(), 8);

        // Merge rules applied in order:
        // "hello" -> ["h","e","l","l","o"]
        // merge h+e  -> ["he","l","l","o"]
        // merge l+l  -> ["he","ll","o"]
        // merge ll+o -> ["he","llo"]
        // merge he+llo -> ["hello"]
        let merges = vec![
            ("h".to_string(), "e".to_string()),    // h + e -> he
            ("l".to_string(), "l".to_string()),    // l + l -> ll
            ("ll".to_string(), "o".to_string()),   // ll + o -> llo
            ("he".to_string(), "llo".to_string()), // he + llo -> hello
        ];

        SimpleBpe::from_data(vocab, merges)
    }

    #[test]
    fn test_bpe_encode_decode() {
        let tok = test_bpe();
        let enc = tok.encode("hello").unwrap();
        assert_eq!(enc.ids, vec![8]); // "hello" = 8

        let decoded = tok.decode(&enc.ids).unwrap();
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_bpe_partial_merge() {
        let tok = test_bpe();
        // "helo" -> ["h","e","l","o"]
        // merge h+e -> ["he","l","o"]
        // no more applicable merges
        let enc = tok.encode("helo").unwrap();
        assert_eq!(enc.ids, vec![5, 3, 4]); // he=5, l=3, o=4
    }

    #[test]
    fn test_bpe_unknown_char() {
        let tok = test_bpe();
        let enc = tok.encode("x").unwrap();
        assert_eq!(enc.ids, vec![0]); // UNK
    }

    #[test]
    fn test_bpe_vocab_size() {
        let tok = test_bpe();
        assert_eq!(tok.vocab_size(), 9);
    }

    #[test]
    fn test_bpe_attention_mask() {
        let tok = test_bpe();
        let enc = tok.encode("hello").unwrap();
        assert_eq!(enc.attention_mask, vec![1]); // single merged token
    }

    #[test]
    fn test_bpe_trait_object() {
        let tok: Box<dyn Tokenizer> = Box::new(test_bpe());
        let enc = tok.encode("hello").unwrap();
        assert!(!enc.ids.is_empty());
    }
}
