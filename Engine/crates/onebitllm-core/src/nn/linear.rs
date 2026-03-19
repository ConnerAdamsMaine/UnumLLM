use ndarray::{Array, Ix2, IxDyn};
use rand::Rng;

use super::module::{Module, Parameter};
use crate::error::OneBitError;
use crate::quant::QuantConfig;
use crate::tensor::PackedTensor;
use crate::Result;

/// A quantized linear layer (BitLinear).
///
/// Stores full-precision latent weights for training (updated by the optimizer).
/// On each forward pass, weights are quantized to ternary {-1, 0, +1}.
/// During inference, pre-packed ternary weights are used directly.
pub struct QuantizedLinear {
    /// Full-precision latent weights (out_features x in_features).
    weight: Parameter,
    /// Optional bias (out_features,).
    bias: Option<Parameter>,
    /// Quantization configuration.
    quant_config: QuantConfig,
    /// Cached packed weights for inference.
    packed_cache: Option<PackedTensor>,
    in_features: usize,
    out_features: usize,
}

impl QuantizedLinear {
    /// Create a new quantized linear layer with random initialization.
    pub fn new(in_features: usize, out_features: usize, bias: bool, config: QuantConfig) -> Self {
        let mut rng = rand::thread_rng();
        // Kaiming uniform initialization
        let bound = (1.0 / in_features as f32).sqrt();
        let weight_data: Vec<f32> = (0..out_features * in_features)
            .map(|_| rng.gen_range(-bound..bound))
            .collect();
        let weight = Parameter::new(
            "weight",
            Array::from_shape_vec(IxDyn(&[out_features, in_features]), weight_data).unwrap(),
        );

        let bias = if bias {
            let bias_data: Vec<f32> = (0..out_features)
                .map(|_| rng.gen_range(-bound..bound))
                .collect();
            Some(Parameter::new(
                "bias",
                Array::from_shape_vec(IxDyn(&[out_features]), bias_data).unwrap(),
            ))
        } else {
            None
        };

        Self {
            weight,
            bias,
            quant_config: config,
            packed_cache: None,
            in_features,
            out_features,
        }
    }

    /// Create from existing weight data.
    pub fn from_weights(
        weight: Array<f32, IxDyn>,
        bias: Option<Array<f32, IxDyn>>,
        config: QuantConfig,
    ) -> Result<Self> {
        if weight.ndim() != 2 {
            return Err(OneBitError::ShapeMismatch {
                expected: vec![0, 0],
                got: weight.shape().to_vec(),
            });
        }
        let out_features = weight.shape()[0];
        let in_features = weight.shape()[1];

        Ok(Self {
            weight: Parameter::new("weight", weight),
            bias: bias.map(|b| Parameter::new("bias", b)),
            quant_config: config,
            packed_cache: None,
            in_features,
            out_features,
        })
    }

    /// Quantize current weights and cache the packed representation.
    pub fn quantize_weights(&mut self) {
        self.packed_cache = Some(PackedTensor::from_ndarray(
            &self.weight.data,
            &self.quant_config,
        ));
    }

    /// Get the current packed (quantized) weights, if cached.
    pub fn packed_weights(&self) -> Option<&PackedTensor> {
        self.packed_cache.as_ref()
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Module for QuantizedLinear {
    fn forward_inference(&self, input: &Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>> {
        // Quantize weights to ternary for inference
        let packed = match &self.packed_cache {
            Some(p) => p.clone(),
            None => PackedTensor::from_ndarray(&self.weight.data, &self.quant_config),
        };

        // input shape: (..., in_features)
        // weight shape: (out_features, in_features)
        // output shape: (..., out_features)
        let input_shape = input.shape().to_vec();
        let batch_dims = &input_shape[..input_shape.len() - 1];
        let in_feat = *input_shape.last().unwrap();

        if in_feat != self.in_features {
            return Err(OneBitError::ShapeMismatch {
                expected: vec![self.in_features],
                got: vec![in_feat],
            });
        }

        // Flatten to 2D: (batch, in_features)
        let batch_size: usize = batch_dims.iter().product();
        let input_2d = input
            .clone()
            .into_shape_with_order(ndarray::IxDyn(&[batch_size, in_feat]))
            .map_err(|e| OneBitError::TensorOp(e.to_string()))?;

        // Keep weights packed throughout inference instead of materializing
        // a dense dequantized matrix for every forward call.
        let input_2d_concrete = input_2d
            .into_dimensionality::<Ix2>()
            .map_err(|e| OneBitError::TensorOp(e.to_string()))?;
        let mut output = packed
            .matmul_dense_left_transposed(&input_2d_concrete.into_dyn())?
            .into_dimensionality::<Ix2>()
            .map_err(|e| OneBitError::TensorOp(e.to_string()))?;

        // Add bias
        if let Some(bias) = &self.bias {
            let bias_1d = bias.data.as_slice().unwrap();
            for mut row in output.rows_mut() {
                for (o, &b) in row.iter_mut().zip(bias_1d.iter()) {
                    *o += b;
                }
            }
        }

        // Reshape to (..., out_features)
        let mut out_shape = batch_dims.to_vec();
        out_shape.push(self.out_features);
        let result = output
            .into_dyn()
            .into_shape_with_order(IxDyn(&out_shape))
            .map_err(|e| OneBitError::TensorOp(e.to_string()))?;

        Ok(result)
    }

    fn parameters(&self) -> Vec<&Parameter> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn name(&self) -> &str {
        "QuantizedLinear"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear_creation() {
        let linear = QuantizedLinear::new(4, 3, true, QuantConfig::per_tensor());
        assert_eq!(linear.in_features(), 4);
        assert_eq!(linear.out_features(), 3);
        assert_eq!(linear.num_parameters(), 4 * 3 + 3); // weight + bias
    }

    #[test]
    fn test_linear_no_bias() {
        let linear = QuantizedLinear::new(4, 3, false, QuantConfig::per_tensor());
        assert_eq!(linear.num_parameters(), 4 * 3);
    }

    #[test]
    fn test_linear_forward_shape() {
        let linear = QuantizedLinear::new(4, 3, false, QuantConfig::per_tensor());
        let input = Array::from_elem(IxDyn(&[2, 4]), 1.0f32);
        let output = linear.forward_inference(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_linear_forward_3d() {
        let linear = QuantizedLinear::new(4, 3, false, QuantConfig::per_tensor());
        let input = Array::from_elem(IxDyn(&[2, 5, 4]), 1.0f32);
        let output = linear.forward_inference(&input).unwrap();
        assert_eq!(output.shape(), &[2, 5, 3]);
    }

    #[test]
    fn test_linear_forward_with_bias() {
        // Weight = identity-like, bias = [1, 0]
        let weight = array![[1.0f32, 0.0], [0.0, 1.0]].into_dyn();
        let bias = array![1.0f32, 0.0].into_dyn();
        let linear =
            QuantizedLinear::from_weights(weight, Some(bias), QuantConfig::per_tensor()).unwrap();

        let input = array![[1.0f32, 2.0]].into_dyn();
        let output = linear.forward_inference(&input).unwrap();
        // Quantized weights will be ternary, so exact values differ
        assert_eq!(output.shape(), &[1, 2]);
    }

    #[test]
    fn test_quantize_and_cache() {
        let mut linear = QuantizedLinear::new(4, 3, false, QuantConfig::per_tensor());
        assert!(linear.packed_weights().is_none());

        linear.quantize_weights();
        assert!(linear.packed_weights().is_some());
    }

    #[test]
    fn test_linear_dimension_mismatch() {
        let linear = QuantizedLinear::new(4, 3, false, QuantConfig::per_tensor());
        let input = Array::from_elem(IxDyn(&[2, 5]), 1.0f32); // wrong input dim
        assert!(linear.forward_inference(&input).is_err());
    }
}
