use ndarray::{Array, IxDyn};

use crate::Result;
use crate::error::OneBitError;
use crate::nn::Module;
use crate::tokenizer::Tokenizer;
use super::sampler::{Sampler, SamplingConfig};

/// Configuration for text generation.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Sampling configuration.
    pub sampling: SamplingConfig,
    /// Token IDs that signal end-of-sequence.
    pub stop_tokens: Vec<u32>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sampling: SamplingConfig::default(),
            stop_tokens: Vec::new(),
        }
    }
}

/// Token-by-token streaming generator.
///
/// Generates text by repeatedly running the model's forward pass
/// and sampling the next token. Yields tokens one at a time.
pub struct Generator<'a> {
    model: &'a dyn Module,
    tokenizer: &'a dyn Tokenizer,
    config: GenerateConfig,
}

impl<'a> Generator<'a> {
    pub fn new(
        model: &'a dyn Module,
        tokenizer: &'a dyn Tokenizer,
        config: GenerateConfig,
    ) -> Self {
        Self {
            model,
            tokenizer,
            config,
        }
    }

    /// Generate tokens as a collected string (blocking).
    pub fn generate_all(&self, prompt: &str) -> Result<String> {
        let encoding = self.tokenizer.encode(prompt)?;
        let mut input_ids = encoding.ids;
        let mut sampler = Sampler::new(self.config.sampling.clone());
        let mut generated_tokens = Vec::new();

        for _ in 0..self.config.max_new_tokens {
            // Build input tensor: (1, seq_len)
            let input_f32: Vec<f32> = input_ids.iter().map(|&id| id as f32).collect();
            let seq_len = input_ids.len();
            let input_tensor =
                Array::from_shape_vec(IxDyn(&[1, seq_len]), input_f32)
                    .map_err(|e| OneBitError::Inference(e.to_string()))?;

            // Forward pass
            let output = self.model.forward_inference(&input_tensor)?;

            // Get logits for the last position: (1, seq_len, vocab) or (1, vocab)
            let output_shape = output.shape().to_vec();
            let logits = if output_shape.len() == 3 {
                // (batch, seq, vocab) -> take last position
                let vocab_size = output_shape[2];
                let last_pos = output_shape[1] - 1;
                let mut last_logits = Array::zeros(IxDyn(&[vocab_size]));
                for v in 0..vocab_size {
                    last_logits[v] = output[[0, last_pos, v]];
                }
                last_logits
            } else if output_shape.len() == 2 {
                // (batch, vocab)
                let vocab_size = output_shape[1];
                let mut last_logits = Array::zeros(IxDyn(&[vocab_size]));
                for v in 0..vocab_size {
                    last_logits[v] = output[[0, v]];
                }
                last_logits
            } else {
                return Err(OneBitError::Inference(format!(
                    "Unexpected output shape: {output_shape:?}"
                )));
            };

            // Sample next token
            let next_token = sampler.sample_with_history(&logits, &generated_tokens);

            // Check for stop token
            if self.config.stop_tokens.contains(&next_token) {
                break;
            }

            generated_tokens.push(next_token);
            input_ids.push(next_token);
        }

        self.tokenizer.decode(&generated_tokens)
    }

    /// Generate tokens as an iterator (streaming).
    pub fn generate_stream(
        &self,
        prompt: &str,
    ) -> Result<GenerateStream<'_>> {
        let encoding = self.tokenizer.encode(prompt)?;
        let sampler = Sampler::new(self.config.sampling.clone());

        Ok(GenerateStream {
            model: self.model,
            tokenizer: self.tokenizer,
            config: &self.config,
            input_ids: encoding.ids,
            generated_ids: Vec::new(),
            sampler,
            done: false,
        })
    }
}

/// Streaming token iterator.
///
/// Each call to `next()` runs one forward pass and returns
/// the decoded token string.
pub struct GenerateStream<'a> {
    model: &'a dyn Module,
    tokenizer: &'a dyn Tokenizer,
    config: &'a GenerateConfig,
    input_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    sampler: Sampler,
    done: bool,
}

impl<'a> GenerateStream<'a> {
    /// Number of tokens generated so far.
    pub fn tokens_generated(&self) -> usize {
        self.generated_ids.len()
    }

    /// All generated token IDs so far.
    pub fn generated_ids(&self) -> &[u32] {
        &self.generated_ids
    }
}

impl Iterator for GenerateStream<'_> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if self.generated_ids.len() >= self.config.max_new_tokens {
            self.done = true;
            return None;
        }

        // Build input tensor
        let input_f32: Vec<f32> = self.input_ids.iter().map(|&id| id as f32).collect();
        let seq_len = self.input_ids.len();
        let input_tensor = match Array::from_shape_vec(IxDyn(&[1, seq_len]), input_f32) {
            Ok(t) => t,
            Err(e) => return Some(Err(OneBitError::Inference(e.to_string()))),
        };

        // Forward pass
        let output = match self.model.forward_inference(&input_tensor) {
            Ok(o) => o,
            Err(e) => return Some(Err(e)),
        };

        // Extract last-position logits
        let output_shape = output.shape().to_vec();
        let logits = if output_shape.len() == 3 {
            let vocab_size = output_shape[2];
            let last_pos = output_shape[1] - 1;
            let mut last_logits = Array::zeros(IxDyn(&[vocab_size]));
            for v in 0..vocab_size {
                last_logits[v] = output[[0, last_pos, v]];
            }
            last_logits
        } else if output_shape.len() == 2 {
            let vocab_size = output_shape[1];
            let mut last_logits = Array::zeros(IxDyn(&[vocab_size]));
            for v in 0..vocab_size {
                last_logits[v] = output[[0, v]];
            }
            last_logits
        } else {
            return Some(Err(OneBitError::Inference(format!(
                "Unexpected output shape: {output_shape:?}"
            ))));
        };

        // Sample
        let next_token = self.sampler.sample_with_history(&logits, &self.generated_ids);

        // Check stop
        if self.config.stop_tokens.contains(&next_token) {
            self.done = true;
            return None;
        }

        self.generated_ids.push(next_token);
        self.input_ids.push(next_token);

        // Decode single token
        match self.tokenizer.decode(&[next_token]) {
            Ok(s) => Some(Ok(s)),
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::module::Parameter;

    // A trivial model that returns random logits for testing
    struct DummyModel {
        vocab_size: usize,
    }

    impl Module for DummyModel {
        fn forward_inference(&self, input: &Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>> {
            let batch = input.shape()[0];
            let seq_len = input.shape()[1];
            // Return (batch, seq_len, vocab_size) with deterministic logits
            let mut output =
                Array::zeros(IxDyn(&[batch, seq_len, self.vocab_size]));
            // Make token 2 always have the highest logit
            for b in 0..batch {
                for s in 0..seq_len {
                    output[[b, s, 2]] = 10.0;
                }
            }
            Ok(output)
        }

        fn parameters(&self) -> Vec<&Parameter> {
            Vec::new()
        }

        fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
            Vec::new()
        }

        fn name(&self) -> &str {
            "DummyModel"
        }
    }

    struct DummyTokenizer;

    impl Tokenizer for DummyTokenizer {
        fn encode(&self, text: &str) -> Result<crate::tokenizer::Encoding> {
            let ids: Vec<u32> = text.bytes().map(|b| b as u32).collect();
            let tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
            let attention_mask = vec![1u32; ids.len()];
            Ok(crate::tokenizer::Encoding {
                ids,
                tokens,
                attention_mask,
            })
        }

        fn decode(&self, ids: &[u32]) -> Result<String> {
            Ok(ids
                .iter()
                .filter_map(|&id| char::from_u32(id))
                .collect())
        }

        fn vocab_size(&self) -> usize {
            256
        }
    }

    #[test]
    fn test_generate_config_default() {
        let config = GenerateConfig::default();
        assert_eq!(config.max_new_tokens, 256);
        assert!(config.stop_tokens.is_empty());
    }

    #[test]
    fn test_generator_greedy() {
        let model = DummyModel { vocab_size: 10 };
        let tokenizer = DummyTokenizer;
        let config = GenerateConfig {
            max_new_tokens: 5,
            sampling: SamplingConfig {
                temperature: 0.0,
                seed: Some(42),
                ..Default::default()
            },
            stop_tokens: Vec::new(),
        };

        let generator = Generator::new(&model, &tokenizer, config);
        let result = generator.generate_all("hi").unwrap();
        // DummyModel always gives highest logit to token 2 (STX in ASCII)
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_generator_stop_token() {
        let model = DummyModel { vocab_size: 10 };
        let tokenizer = DummyTokenizer;
        let config = GenerateConfig {
            max_new_tokens: 100,
            sampling: SamplingConfig {
                temperature: 0.0,
                seed: Some(42),
                ..Default::default()
            },
            stop_tokens: vec![2], // Token 2 is stop token
        };

        let generator = Generator::new(&model, &tokenizer, config);
        let result = generator.generate_all("hi").unwrap();
        // Should stop immediately since token 2 is the stop token
        assert!(result.is_empty());
    }

    #[test]
    fn test_generate_stream() {
        let model = DummyModel { vocab_size: 10 };
        let tokenizer = DummyTokenizer;
        let config = GenerateConfig {
            max_new_tokens: 3,
            sampling: SamplingConfig {
                temperature: 0.0,
                seed: Some(42),
                ..Default::default()
            },
            stop_tokens: Vec::new(),
        };

        let generator = Generator::new(&model, &tokenizer, config);
        let stream = generator.generate_stream("hi").unwrap();

        let tokens: Vec<String> = stream.map(|r| r.unwrap()).collect();
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn test_generate_stream_stop() {
        let model = DummyModel { vocab_size: 10 };
        let tokenizer = DummyTokenizer;
        let config = GenerateConfig {
            max_new_tokens: 100,
            sampling: SamplingConfig {
                temperature: 0.0,
                seed: Some(42),
                ..Default::default()
            },
            stop_tokens: vec![2],
        };

        let generator = Generator::new(&model, &tokenizer, config);
        let stream = generator.generate_stream("hi").unwrap();

        let tokens: Vec<String> = stream.map(|r| r.unwrap()).collect();
        assert_eq!(tokens.len(), 0);
    }
}
