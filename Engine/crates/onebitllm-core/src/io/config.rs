//! Model configuration for serialization and deserialization.
//!
//! Describes the architecture of a model so it can be reconstructed
//! from saved weights without additional metadata.

use std::collections::HashMap;
use std::io::{Read, Write};

use crate::Result;

/// Model architecture configuration.
///
/// Describes all hyperparameters needed to reconstruct a model's
/// layer structure (without the weight data itself).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// Model architecture name (e.g., "bitnet-b1.58", "llama-1bit").
    pub architecture: String,
    /// Hidden dimension (embedding size).
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA; equals num_attention_heads for MHA).
    pub num_kv_heads: usize,
    /// Intermediate/FFN dimension.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// RMS norm epsilon.
    pub rms_norm_eps: f32,
    /// Activation function name (e.g., "silu", "gelu", "swiglu").
    pub activation: String,
    /// Positional encoding type (e.g., "rope", "alibi", "learned").
    pub positional_encoding: String,
    /// RoPE base frequency (if using RoPE).
    pub rope_theta: f32,
    /// Whether to use bias in linear layers.
    pub use_bias: bool,
    /// Quantization group size (0 for per-tensor).
    pub quant_group_size: usize,
    /// Additional arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: "bitnet-b1.58".into(),
            hidden_size: 768,
            num_layers: 12,
            num_attention_heads: 12,
            num_kv_heads: 12,
            intermediate_size: 2048,
            vocab_size: 32000,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            activation: "silu".into(),
            positional_encoding: "rope".into(),
            rope_theta: 10000.0,
            use_bias: false,
            quant_group_size: 0,
            metadata: HashMap::new(),
        }
    }
}

impl ModelConfig {
    /// Save config to a writer as JSON.
    pub fn save_json<W: Write>(&self, mut w: W) -> Result<()> {
        let json = self.to_json_string()?;
        w.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Load config from a reader as JSON.
    pub fn load_json<R: Read>(mut r: R) -> Result<Self> {
        let mut buf = String::new();
        r.read_to_string(&mut buf)?;
        Self::from_json_str(&buf)
    }

    /// Serialize to a JSON string (without serde_json dependency — manual).
    fn to_json_string(&self) -> Result<String> {
        let mut s = String::from("{\n");
        s.push_str(&format!("  \"architecture\": {},\n", json_str(&self.architecture)));
        s.push_str(&format!("  \"hidden_size\": {},\n", self.hidden_size));
        s.push_str(&format!("  \"num_layers\": {},\n", self.num_layers));
        s.push_str(&format!("  \"num_attention_heads\": {},\n", self.num_attention_heads));
        s.push_str(&format!("  \"num_kv_heads\": {},\n", self.num_kv_heads));
        s.push_str(&format!("  \"intermediate_size\": {},\n", self.intermediate_size));
        s.push_str(&format!("  \"vocab_size\": {},\n", self.vocab_size));
        s.push_str(&format!("  \"max_seq_len\": {},\n", self.max_seq_len));
        s.push_str(&format!("  \"rms_norm_eps\": {},\n", self.rms_norm_eps));
        s.push_str(&format!("  \"activation\": {},\n", json_str(&self.activation)));
        s.push_str(&format!("  \"positional_encoding\": {},\n", json_str(&self.positional_encoding)));
        s.push_str(&format!("  \"rope_theta\": {},\n", self.rope_theta));
        s.push_str(&format!("  \"use_bias\": {},\n", self.use_bias));
        s.push_str(&format!("  \"quant_group_size\": {}", self.quant_group_size));

        if !self.metadata.is_empty() {
            s.push_str(",\n  \"metadata\": {\n");
            let entries: Vec<_> = self.metadata.iter().collect();
            for (i, (k, v)) in entries.iter().enumerate() {
                s.push_str(&format!("    {}: {}", json_str(k), json_str(v)));
                if i + 1 < entries.len() {
                    s.push(',');
                }
                s.push('\n');
            }
            s.push_str("  }");
        }

        s.push_str("\n}\n");
        Ok(s)
    }

    /// Parse from a JSON string (minimal parser, no serde required).
    pub fn from_json_str(s: &str) -> Result<Self> {
        let mut config = Self::default();
        let s = s.trim();

        // Simple key-value extraction from JSON
        if let Some(v) = extract_json_str(s, "architecture") {
            config.architecture = v;
        }
        if let Some(v) = extract_json_usize(s, "hidden_size") {
            config.hidden_size = v;
        }
        if let Some(v) = extract_json_usize(s, "num_layers") {
            config.num_layers = v;
        }
        if let Some(v) = extract_json_usize(s, "num_attention_heads") {
            config.num_attention_heads = v;
        }
        if let Some(v) = extract_json_usize(s, "num_kv_heads") {
            config.num_kv_heads = v;
        }
        if let Some(v) = extract_json_usize(s, "intermediate_size") {
            config.intermediate_size = v;
        }
        if let Some(v) = extract_json_usize(s, "vocab_size") {
            config.vocab_size = v;
        }
        if let Some(v) = extract_json_usize(s, "max_seq_len") {
            config.max_seq_len = v;
        }
        if let Some(v) = extract_json_f32(s, "rms_norm_eps") {
            config.rms_norm_eps = v;
        }
        if let Some(v) = extract_json_str(s, "activation") {
            config.activation = v;
        }
        if let Some(v) = extract_json_str(s, "positional_encoding") {
            config.positional_encoding = v;
        }
        if let Some(v) = extract_json_f32(s, "rope_theta") {
            config.rope_theta = v;
        }
        if let Some(v) = extract_json_bool(s, "use_bias") {
            config.use_bias = v;
        }
        if let Some(v) = extract_json_usize(s, "quant_group_size") {
            config.quant_group_size = v;
        }

        // Parse metadata object (simple flat string->string)
        if let Some(meta_start) = s.find("\"metadata\"") {
            if let Some(brace_start) = s[meta_start..].find('{') {
                let meta_substr = &s[meta_start + brace_start..];
                if let Some(brace_end) = meta_substr.find('}') {
                    let inner = &meta_substr[1..brace_end];
                    // Parse "key": "value" pairs
                    let mut pos = 0;
                    while pos < inner.len() {
                        if let Some(key_start) = inner[pos..].find('"') {
                            let key_start = pos + key_start + 1;
                            if let Some(key_end) = inner[key_start..].find('"') {
                                let key = inner[key_start..key_start + key_end].to_string();
                                let after_key = key_start + key_end + 1;
                                if let Some(val_start) = inner[after_key..].find('"') {
                                    let val_start = after_key + val_start + 1;
                                    if let Some(val_end) = inner[val_start..].find('"') {
                                        let val = inner[val_start..val_start + val_end].to_string();
                                        config.metadata.insert(key, val);
                                        pos = val_start + val_end + 1;
                                        continue;
                                    }
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }

        Ok(config)
    }

    /// Save config to a writer as YAML (requires `yaml-config` feature).
    #[cfg(feature = "yaml-config")]
    pub fn save_yaml<W: Write>(&self, mut w: W) -> Result<()> {
        let yaml = self.to_yaml_string();
        w.write_all(yaml.as_bytes())?;
        Ok(())
    }

    /// Convert to a YAML string (requires `yaml-config` feature).
    #[cfg(feature = "yaml-config")]
    fn to_yaml_string(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("architecture: {}\n", &self.architecture));
        s.push_str(&format!("hidden_size: {}\n", self.hidden_size));
        s.push_str(&format!("num_layers: {}\n", self.num_layers));
        s.push_str(&format!("num_attention_heads: {}\n", self.num_attention_heads));
        s.push_str(&format!("num_kv_heads: {}\n", self.num_kv_heads));
        s.push_str(&format!("intermediate_size: {}\n", self.intermediate_size));
        s.push_str(&format!("vocab_size: {}\n", self.vocab_size));
        s.push_str(&format!("max_seq_len: {}\n", self.max_seq_len));
        s.push_str(&format!("rms_norm_eps: {}\n", self.rms_norm_eps));
        s.push_str(&format!("activation: {}\n", &self.activation));
        s.push_str(&format!("positional_encoding: {}\n", &self.positional_encoding));
        s.push_str(&format!("rope_theta: {}\n", self.rope_theta));
        s.push_str(&format!("use_bias: {}\n", self.use_bias));
        s.push_str(&format!("quant_group_size: {}\n", self.quant_group_size));
        if !self.metadata.is_empty() {
            s.push_str("metadata:\n");
            for (k, v) in &self.metadata {
                s.push_str(&format!("  {k}: {v}\n"));
            }
        }
        s
    }

    /// Compute the head dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Escape a string for JSON output.
fn json_str(s: &str) -> String {
    let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

/// Extract a string value for a key from JSON text.
fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let idx = json.find(&pattern)?;
    let after = &json[idx + pattern.len()..];
    // Find colon, then opening quote
    let colon = after.find(':')?;
    let after_colon = after[colon + 1..].trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    let start = 1; // skip opening quote
    let end = after_colon[start..].find('"')?;
    Some(after_colon[start..start + end].to_string())
}

/// Extract a usize value for a key from JSON text.
fn extract_json_usize(json: &str, key: &str) -> Option<usize> {
    let pattern = format!("\"{key}\"");
    let idx = json.find(&pattern)?;
    let after = &json[idx + pattern.len()..];
    let colon = after.find(':')?;
    let after_colon = after[colon + 1..].trim_start();
    // Parse number until non-digit
    let num_str: String = after_colon.chars().take_while(|c| c.is_ascii_digit()).collect();
    num_str.parse().ok()
}

/// Extract an f32 value for a key from JSON text.
fn extract_json_f32(json: &str, key: &str) -> Option<f32> {
    let pattern = format!("\"{key}\"");
    let idx = json.find(&pattern)?;
    let after = &json[idx + pattern.len()..];
    let colon = after.find(':')?;
    let after_colon = after[colon + 1..].trim_start();
    let num_str: String = after_colon
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e' || *c == 'E' || *c == '+')
        .collect();
    num_str.parse().ok()
}

/// Extract a bool value for a key from JSON text.
fn extract_json_bool(json: &str, key: &str) -> Option<bool> {
    let pattern = format!("\"{key}\"");
    let idx = json.find(&pattern)?;
    let after = &json[idx + pattern.len()..];
    let colon = after.find(':')?;
    let after_colon = after[colon + 1..].trim_start();
    if after_colon.starts_with("true") {
        Some(true)
    } else if after_colon.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_default_config() {
        let config = ModelConfig::default();
        assert_eq!(config.architecture, "bitnet-b1.58");
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_json_roundtrip() {
        let config = ModelConfig {
            architecture: "test-model".into(),
            hidden_size: 512,
            num_layers: 6,
            num_attention_heads: 8,
            num_kv_heads: 4,
            intermediate_size: 1024,
            vocab_size: 16000,
            max_seq_len: 1024,
            rms_norm_eps: 1e-6,
            activation: "gelu".into(),
            positional_encoding: "alibi".into(),
            rope_theta: 500000.0,
            use_bias: true,
            quant_group_size: 64,
            metadata: HashMap::new(),
        };

        let mut buf = Vec::new();
        config.save_json(&mut buf).unwrap();

        let loaded = ModelConfig::load_json(Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.architecture, "test-model");
        assert_eq!(loaded.hidden_size, 512);
        assert_eq!(loaded.num_layers, 6);
        assert_eq!(loaded.num_attention_heads, 8);
        assert_eq!(loaded.num_kv_heads, 4);
        assert_eq!(loaded.intermediate_size, 1024);
        assert_eq!(loaded.vocab_size, 16000);
        assert_eq!(loaded.max_seq_len, 1024);
        assert_eq!(loaded.activation, "gelu");
        assert_eq!(loaded.positional_encoding, "alibi");
        assert!(loaded.use_bias);
        assert_eq!(loaded.quant_group_size, 64);
    }

    #[test]
    fn test_json_with_metadata() {
        let mut config = ModelConfig::default();
        config.metadata.insert("author".into(), "test".into());
        config.metadata.insert("version".into(), "1.0".into());

        let mut buf = Vec::new();
        config.save_json(&mut buf).unwrap();

        let loaded = ModelConfig::load_json(Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.metadata.get("author").unwrap(), "test");
        assert_eq!(loaded.metadata.get("version").unwrap(), "1.0");
    }

    #[test]
    fn test_head_dim() {
        let config = ModelConfig {
            hidden_size: 1024,
            num_attention_heads: 16,
            ..Default::default()
        };
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_json_helpers() {
        assert_eq!(json_str("hello"), "\"hello\"");
        assert_eq!(json_str("he\"llo"), "\"he\\\"llo\"");

        let json = r#"{"key": 42, "name": "test"}"#;
        assert_eq!(extract_json_usize(json, "key"), Some(42));
        assert_eq!(extract_json_str(json, "name"), Some("test".to_string()));
    }
}
