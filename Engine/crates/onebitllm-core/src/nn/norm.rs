use ndarray::{Array, IxDyn};

use crate::Result;
use crate::error::OneBitError;
use super::module::{Module, Parameter};

/// RMSNorm (Root Mean Square Layer Normalization).
///
/// Used in LLaMA, BitNet, and many modern LLMs. Normalizes the input
/// by its RMS value and applies a learnable scale.
///
/// `y = x / rms(x) * weight`, where `rms(x) = sqrt(mean(x^2) + eps)`
pub struct RmsNorm {
    weight: Parameter,
    eps: f32,
    hidden_size: usize,
}

impl RmsNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        let weight = Parameter::new(
            "weight",
            Array::from_elem(IxDyn(&[hidden_size]), 1.0f32),
        );
        Self {
            weight,
            eps,
            hidden_size,
        }
    }
}

impl Module for RmsNorm {
    fn forward_inference(&self, input: &Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>> {
        // Input: (..., hidden_size)
        let shape = input.shape().to_vec();
        let last_dim = *shape.last().unwrap();

        if last_dim != self.hidden_size {
            return Err(OneBitError::ShapeMismatch {
                expected: vec![self.hidden_size],
                got: vec![last_dim],
            });
        }

        // Flatten to 2D: (batch, hidden_size)
        let batch_size: usize = shape[..shape.len() - 1].iter().product();
        let flat = input
            .clone()
            .into_shape_with_order(IxDyn(&[batch_size, self.hidden_size]))
            .map_err(|e| OneBitError::TensorOp(e.to_string()))?;

        let weight_slice = self.weight.data.as_slice().unwrap();

        let mut output = Array::zeros(flat.raw_dim());
        for (i, row) in flat.rows().into_iter().enumerate() {
            // Compute RMS
            let sq_mean: f32 = row.iter().map(|&x| x * x).sum::<f32>() / self.hidden_size as f32;
            let rms = (sq_mean + self.eps).sqrt();

            for (j, &x) in row.iter().enumerate() {
                output[[i, j]] = x / rms * weight_slice[j];
            }
        }

        output
            .into_shape_with_order(IxDyn(&shape))
            .map_err(|e| OneBitError::TensorOp(e.to_string()))
    }

    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.weight]
    }

    fn name(&self) -> &str {
        "RmsNorm"
    }
}

/// Standard Layer Normalization.
///
/// `y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias`
pub struct LayerNorm {
    weight: Parameter,
    bias: Parameter,
    eps: f32,
    hidden_size: usize,
}

impl LayerNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        let weight = Parameter::new(
            "weight",
            Array::from_elem(IxDyn(&[hidden_size]), 1.0f32),
        );
        let bias = Parameter::new(
            "bias",
            Array::from_elem(IxDyn(&[hidden_size]), 0.0f32),
        );
        Self {
            weight,
            bias,
            eps,
            hidden_size,
        }
    }
}

impl Module for LayerNorm {
    fn forward_inference(&self, input: &Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>> {
        let shape = input.shape().to_vec();
        let last_dim = *shape.last().unwrap();

        if last_dim != self.hidden_size {
            return Err(OneBitError::ShapeMismatch {
                expected: vec![self.hidden_size],
                got: vec![last_dim],
            });
        }

        let batch_size: usize = shape[..shape.len() - 1].iter().product();
        let flat = input
            .clone()
            .into_shape_with_order(IxDyn(&[batch_size, self.hidden_size]))
            .map_err(|e| OneBitError::TensorOp(e.to_string()))?;

        let weight_slice = self.weight.data.as_slice().unwrap();
        let bias_slice = self.bias.data.as_slice().unwrap();

        let mut output = Array::zeros(flat.raw_dim());
        for (i, row) in flat.rows().into_iter().enumerate() {
            let mean: f32 = row.iter().sum::<f32>() / self.hidden_size as f32;
            let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>()
                / self.hidden_size as f32;
            let std = (var + self.eps).sqrt();

            for (j, &x) in row.iter().enumerate() {
                output[[i, j]] = (x - mean) / std * weight_slice[j] + bias_slice[j];
            }
        }

        output
            .into_shape_with_order(IxDyn(&shape))
            .map_err(|e| OneBitError::TensorOp(e.to_string()))
    }

    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn name(&self) -> &str {
        "LayerNorm"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_rms_norm_shape() {
        let norm = RmsNorm::new(4, 1e-6);
        let input = Array::from_elem(IxDyn(&[2, 3, 4]), 1.0f32);
        let output = norm.forward_inference(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_rms_norm_unit_rms() {
        let norm = RmsNorm::new(4, 1e-6);
        let input = array![1.0f32, 2.0, 3.0, 4.0].into_dyn();
        let output = norm.forward_inference(&input).unwrap();

        // After RMSNorm (with weight=1), output should have RMS ≈ 1
        let rms: f32 = (output.iter().map(|x| x * x).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 0.01, "RMS = {rms}");
    }

    #[test]
    fn test_layer_norm_shape() {
        let norm = LayerNorm::new(4, 1e-5);
        let input = Array::from_elem(IxDyn(&[2, 3, 4]), 1.0f32);
        let output = norm.forward_inference(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_layer_norm_zero_mean_unit_var() {
        let norm = LayerNorm::new(4, 1e-5);
        let input = array![1.0f32, 2.0, 3.0, 4.0].into_dyn();
        let output = norm.forward_inference(&input).unwrap();

        // After LayerNorm (with weight=1, bias=0), output should have mean≈0 and var≈1
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        let var: f32 = output.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / 4.0;
        assert!((mean).abs() < 0.01, "mean = {mean}");
        assert!((var - 1.0).abs() < 0.01, "var = {var}");
    }

    #[test]
    fn test_rms_norm_dimension_mismatch() {
        let norm = RmsNorm::new(4, 1e-6);
        let input = Array::from_elem(IxDyn(&[2, 5]), 1.0f32);
        assert!(norm.forward_inference(&input).is_err());
    }

    #[test]
    fn test_rms_norm_parameters() {
        let norm = RmsNorm::new(4, 1e-6);
        assert_eq!(norm.num_parameters(), 4);
    }

    #[test]
    fn test_layer_norm_parameters() {
        let norm = LayerNorm::new(4, 1e-5);
        assert_eq!(norm.num_parameters(), 8); // weight + bias
    }
}
