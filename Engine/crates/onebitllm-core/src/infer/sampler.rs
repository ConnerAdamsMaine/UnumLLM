use ndarray::{Array, IxDyn};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Configuration for token sampling.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            seed: None,
        }
    }
}

/// Token sampler with configurable strategies.
pub struct Sampler {
    config: SamplingConfig,
    rng: StdRng,
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Self { config, rng }
    }

    /// Sample a token ID from logits (1D array of shape [vocab_size]).
    pub fn sample(&mut self, logits: &Array<f32, IxDyn>) -> u32 {
        let mut logits = logits.clone();

        // Apply temperature
        self.apply_temperature(&mut logits);

        // Apply top-k
        self.apply_top_k(&mut logits);

        // Apply top-p
        self.apply_top_p(&mut logits);

        // Convert to probabilities via softmax
        let probs = softmax_1d(&logits);

        // Sample from distribution
        self.sample_from_probs(&probs)
    }

    /// Sample a token with repetition penalty applied.
    pub fn sample_with_history(
        &mut self,
        logits: &Array<f32, IxDyn>,
        generated_ids: &[u32],
    ) -> u32 {
        let mut logits = logits.clone();

        // Apply repetition penalty
        if let Some(penalty) = self.config.repetition_penalty {
            apply_repetition_penalty(&mut logits, generated_ids, penalty);
        }

        self.apply_temperature(&mut logits);
        self.apply_top_k(&mut logits);
        self.apply_top_p(&mut logits);

        let probs = softmax_1d(&logits);
        self.sample_from_probs(&probs)
    }

    fn apply_temperature(&self, logits: &mut Array<f32, IxDyn>) {
        if self.config.temperature != 1.0 && self.config.temperature > 0.0 {
            logits.mapv_inplace(|x| x / self.config.temperature);
        }
    }

    fn apply_top_k(&self, logits: &mut Array<f32, IxDyn>) {
        if let Some(k) = self.config.top_k {
            let flat = logits.as_slice_mut().unwrap();
            if k >= flat.len() {
                return;
            }

            // Find the k-th largest value
            let mut sorted: Vec<f32> = flat.to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let threshold = sorted[k];

            // Zero out everything below threshold
            for v in flat.iter_mut() {
                if *v < threshold {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
    }

    fn apply_top_p(&self, logits: &mut Array<f32, IxDyn>) {
        if let Some(p) = self.config.top_p {
            let flat = logits.as_slice_mut().unwrap();

            // Sort indices by logit value descending
            let mut indices: Vec<usize> = (0..flat.len()).collect();
            indices.sort_by(|&a, &b| flat[b].partial_cmp(&flat[a]).unwrap());

            // Compute cumulative softmax probability
            let probs = softmax_slice(flat);
            let mut cumulative = 0.0f32;
            let mut cutoff_idx = indices.len();

            for (rank, &idx) in indices.iter().enumerate() {
                cumulative += probs[idx];
                if cumulative > p {
                    cutoff_idx = rank + 1;
                    break;
                }
            }

            // Zero out tokens beyond the nucleus
            for &idx in &indices[cutoff_idx..] {
                flat[idx] = f32::NEG_INFINITY;
            }
        }
    }

    fn sample_from_probs(&mut self, probs: &Array<f32, IxDyn>) -> u32 {
        let flat = probs.as_slice().unwrap();

        // Temperature=0 or very low -> argmax
        if self.config.temperature <= 0.01 {
            return flat
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }

        // Multinomial sampling
        let r: f32 = self.rng.r#gen();
        let mut cumulative = 0.0;
        for (i, &p) in flat.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return i as u32;
            }
        }

        // Fallback to last token
        (flat.len() - 1) as u32
    }
}

fn softmax_1d(logits: &Array<f32, IxDyn>) -> Array<f32, IxDyn> {
    let flat = logits.as_slice().unwrap();
    let probs = softmax_slice(flat);
    Array::from_shape_vec(logits.raw_dim(), probs).unwrap()
}

fn softmax_slice(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    if sum > 0.0 {
        exp.iter().map(|&e| e / sum).collect()
    } else {
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}

fn apply_repetition_penalty(logits: &mut Array<f32, IxDyn>, generated_ids: &[u32], penalty: f32) {
    let flat = logits.as_slice_mut().unwrap();
    for &id in generated_ids {
        let idx = id as usize;
        if idx < flat.len() {
            if flat[idx] > 0.0 {
                flat[idx] /= penalty;
            } else {
                flat[idx] *= penalty;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_argmax_sampling() {
        let config = SamplingConfig {
            temperature: 0.0, // greedy
            seed: Some(42),
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);

        // Logits: [1.0, 5.0, 2.0, 0.5] -> should always pick index 1
        let logits = array![1.0f32, 5.0, 2.0, 0.5].into_dyn();
        let token = sampler.sample(&logits);
        assert_eq!(token, 1);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let config = SamplingConfig {
            temperature: 0.8,
            seed: Some(123),
            ..Default::default()
        };
        let logits = array![1.0f32, 1.0, 1.0, 1.0].into_dyn();

        let mut sampler1 = Sampler::new(config.clone());
        let mut sampler2 = Sampler::new(config);

        let tokens1: Vec<u32> = (0..10).map(|_| sampler1.sample(&logits)).collect();
        let tokens2: Vec<u32> = (0..10).map(|_| sampler2.sample(&logits)).collect();
        assert_eq!(tokens1, tokens2);
    }

    #[test]
    fn test_top_k() {
        let config = SamplingConfig {
            temperature: 0.001, // near-greedy
            top_k: Some(2),
            seed: Some(42),
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);

        // Only top-2 should be considered: indices 1 and 2
        let logits = array![0.1f32, 10.0, 8.0, 0.2, 0.3].into_dyn();
        let token = sampler.sample(&logits);
        assert!(token == 1 || token == 2);
    }

    #[test]
    fn test_top_p() {
        let config = SamplingConfig {
            temperature: 0.001,
            top_p: Some(0.5),
            seed: Some(42),
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);

        // Index 1 has by far the highest logit, should dominate
        let logits = array![0.1f32, 10.0, 0.2, 0.3, 0.1].into_dyn();
        let token = sampler.sample(&logits);
        assert_eq!(token, 1);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = array![5.0f32, 5.0, 5.0].into_dyn();
        apply_repetition_penalty(&mut logits, &[0], 2.0);
        // Index 0 should be penalized (divided by 2 since positive)
        assert!((logits[0] - 2.5).abs() < 1e-6);
        assert!((logits[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let logits = array![1.0f32, 2.0, 3.0].into_dyn();
        let probs = softmax_1d(&logits);

        // Sum should be 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Probabilities should be ordered
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_sampling_config_default() {
        let config = SamplingConfig::default();
        assert_eq!(config.temperature, 1.0);
        assert!(config.top_k.is_none());
        assert!(config.top_p.is_none());
        assert!(config.seed.is_none());
    }
}
