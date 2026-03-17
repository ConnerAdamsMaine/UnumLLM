/// Quantization configuration and parameter computation.
///
/// Supports per-tensor, per-channel (per-row), and per-group quantization
/// with optional zero-points for asymmetric quantization.

/// Granularity of quantization scale computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantGranularity {
    /// One scale for the entire tensor.
    PerTensor,
    /// One scale per output channel (row of weight matrix).
    PerChannel,
    /// One scale per group of `n` consecutive weights.
    PerGroup(usize),
}

/// Configuration for how quantization should be performed.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Granularity of the quantization scales.
    pub granularity: QuantGranularity,
    /// Whether to compute and use zero-points (asymmetric quantization).
    pub use_zero_point: bool,
    /// Whether the scale parameter is learnable during QAT.
    pub learnable_scale: bool,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            granularity: QuantGranularity::PerTensor,
            use_zero_point: false,
            learnable_scale: false,
        }
    }
}

impl QuantConfig {
    /// Create a per-tensor quantization config.
    pub fn per_tensor() -> Self {
        Self::default()
    }

    /// Create a per-channel quantization config.
    pub fn per_channel() -> Self {
        Self {
            granularity: QuantGranularity::PerChannel,
            ..Self::default()
        }
    }

    /// Create a per-group quantization config.
    pub fn per_group(group_size: usize) -> Self {
        Self {
            granularity: QuantGranularity::PerGroup(group_size),
            ..Self::default()
        }
    }
}

/// Computed quantization parameters for a weight tensor.
///
/// Stores the scales (and optionally zero-points) determined
/// by analyzing the weight distribution.
#[derive(Debug, Clone)]
pub struct QuantParams {
    /// Scales: one per quantization group.
    pub scales: Vec<f32>,
    /// Zero-points: one per quantization group (empty if symmetric).
    pub zero_points: Vec<f32>,
    /// Shape of the original weight tensor.
    pub original_shape: Vec<usize>,
    /// Granularity used to compute these parameters.
    pub granularity: QuantGranularity,
}

impl QuantParams {
    /// Compute quantization parameters from an f32 weight slice.
    ///
    /// For ternary quantization, the scale is computed as the mean of
    /// absolute values (absmean) per quantization group.
    ///
    /// `shape` describes the N-dimensional shape of the weight tensor.
    /// For per-channel, `shape[0]` is the number of output channels (rows).
    pub fn compute(weights: &[f32], shape: &[usize], config: &QuantConfig) -> Self {
        let total_elements: usize = shape.iter().product();
        debug_assert_eq!(
            weights.len(),
            total_elements,
            "weights length {} doesn't match shape {:?} (total {})",
            weights.len(),
            shape,
            total_elements
        );

        match config.granularity {
            QuantGranularity::PerTensor => {
                let scale = absmean(weights);
                let zero_point = if config.use_zero_point {
                    mean(weights)
                } else {
                    0.0
                };
                QuantParams {
                    scales: vec![scale],
                    zero_points: if config.use_zero_point {
                        vec![zero_point]
                    } else {
                        Vec::new()
                    },
                    original_shape: shape.to_vec(),
                    granularity: config.granularity,
                }
            }
            QuantGranularity::PerChannel => {
                let num_channels = shape[0];
                let channel_size = total_elements / num_channels;
                let mut scales = Vec::with_capacity(num_channels);
                let mut zero_points = Vec::new();
                if config.use_zero_point {
                    zero_points.reserve(num_channels);
                }

                for c in 0..num_channels {
                    let start = c * channel_size;
                    let end = start + channel_size;
                    let channel_weights = &weights[start..end];
                    scales.push(absmean(channel_weights));
                    if config.use_zero_point {
                        zero_points.push(mean(channel_weights));
                    }
                }

                QuantParams {
                    scales,
                    zero_points,
                    original_shape: shape.to_vec(),
                    granularity: config.granularity,
                }
            }
            QuantGranularity::PerGroup(group_size) => {
                let num_groups = (total_elements + group_size - 1) / group_size;
                let mut scales = Vec::with_capacity(num_groups);
                let mut zero_points = Vec::new();
                if config.use_zero_point {
                    zero_points.reserve(num_groups);
                }

                for g in 0..num_groups {
                    let start = g * group_size;
                    let end = (start + group_size).min(total_elements);
                    let group_weights = &weights[start..end];
                    scales.push(absmean(group_weights));
                    if config.use_zero_point {
                        zero_points.push(mean(group_weights));
                    }
                }

                QuantParams {
                    scales,
                    zero_points,
                    original_shape: shape.to_vec(),
                    granularity: config.granularity,
                }
            }
        }
    }

    /// Get the scale for a specific weight index (flat index into the tensor).
    pub fn scale_for_index(&self, flat_index: usize) -> f32 {
        match self.granularity {
            QuantGranularity::PerTensor => self.scales[0],
            QuantGranularity::PerChannel => {
                let total: usize = self.original_shape.iter().product();
                let channel_size = total / self.original_shape[0];
                let channel = flat_index / channel_size;
                self.scales[channel]
            }
            QuantGranularity::PerGroup(group_size) => {
                let group = flat_index / group_size;
                self.scales[group]
            }
        }
    }

    /// Number of quantization groups.
    pub fn num_groups(&self) -> usize {
        self.scales.len()
    }
}

/// Compute the mean of absolute values.
fn absmean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|v| v.abs()).sum::<f32>() / values.len() as f32
}

/// Compute the arithmetic mean.
fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_per_tensor_symmetric() {
        let weights = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.0];
        let config = QuantConfig::per_tensor();
        let params = QuantParams::compute(&weights, &[6], &config);

        assert_eq!(params.scales.len(), 1);
        assert!(params.zero_points.is_empty());
        // absmean = (1 + 1 + 0.5 + 0.5 + 0 + 0) / 6 = 0.5
        assert!((params.scales[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_per_tensor_with_zero_point() {
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let config = QuantConfig {
            granularity: QuantGranularity::PerTensor,
            use_zero_point: true,
            learnable_scale: false,
        };
        let params = QuantParams::compute(&weights, &[4], &config);

        assert_eq!(params.scales.len(), 1);
        assert_eq!(params.zero_points.len(), 1);
        // mean = (1+2+3+4)/4 = 2.5
        assert!((params.zero_points[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_per_channel() {
        // 2 channels, 3 weights each
        let weights = vec![
            1.0, -1.0, 0.0, // channel 0: absmean = 2/3
            2.0, -2.0, 0.0, // channel 1: absmean = 4/3
        ];
        let config = QuantConfig::per_channel();
        let params = QuantParams::compute(&weights, &[2, 3], &config);

        assert_eq!(params.scales.len(), 2);
        assert!((params.scales[0] - 2.0 / 3.0).abs() < 1e-6);
        assert!((params.scales[1] - 4.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_per_group() {
        let weights = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5];
        let config = QuantConfig::per_group(2);
        let params = QuantParams::compute(&weights, &[6], &config);

        assert_eq!(params.scales.len(), 3); // 6 / 2 = 3 groups
        assert!((params.scales[0] - 1.0).abs() < 1e-6); // absmean(1, -1) = 1
        assert!((params.scales[1] - 2.0).abs() < 1e-6); // absmean(2, -2) = 2
        assert!((params.scales[2] - 0.5).abs() < 1e-6); // absmean(0.5, -0.5) = 0.5
    }

    #[test]
    fn test_scale_for_index_per_tensor() {
        let params = QuantParams {
            scales: vec![0.5],
            zero_points: Vec::new(),
            original_shape: vec![10],
            granularity: QuantGranularity::PerTensor,
        };
        assert_eq!(params.scale_for_index(0), 0.5);
        assert_eq!(params.scale_for_index(9), 0.5);
    }

    #[test]
    fn test_scale_for_index_per_channel() {
        let params = QuantParams {
            scales: vec![1.0, 2.0],
            zero_points: Vec::new(),
            original_shape: vec![2, 4],
            granularity: QuantGranularity::PerChannel,
        };
        // Channel 0: indices 0..4
        assert_eq!(params.scale_for_index(0), 1.0);
        assert_eq!(params.scale_for_index(3), 1.0);
        // Channel 1: indices 4..8
        assert_eq!(params.scale_for_index(4), 2.0);
        assert_eq!(params.scale_for_index(7), 2.0);
    }

    #[test]
    fn test_scale_for_index_per_group() {
        let params = QuantParams {
            scales: vec![1.0, 2.0, 3.0],
            zero_points: Vec::new(),
            original_shape: vec![6],
            granularity: QuantGranularity::PerGroup(2),
        };
        assert_eq!(params.scale_for_index(0), 1.0);
        assert_eq!(params.scale_for_index(1), 1.0);
        assert_eq!(params.scale_for_index(2), 2.0);
        assert_eq!(params.scale_for_index(3), 2.0);
        assert_eq!(params.scale_for_index(4), 3.0);
        assert_eq!(params.scale_for_index(5), 3.0);
    }

    #[test]
    fn test_per_group_partial_last_group() {
        // 5 weights with group size 2 => groups: [0,1], [2,3], [4]
        let weights = vec![1.0, -1.0, 2.0, -2.0, 0.5];
        let config = QuantConfig::per_group(2);
        let params = QuantParams::compute(&weights, &[5], &config);

        assert_eq!(params.scales.len(), 3);
        assert!((params.scales[2] - 0.5).abs() < 1e-6); // last group has 1 element
    }

    #[test]
    fn test_default_config() {
        let config = QuantConfig::default();
        assert_eq!(config.granularity, QuantGranularity::PerTensor);
        assert!(!config.use_zero_point);
        assert!(!config.learnable_scale);
    }
}
