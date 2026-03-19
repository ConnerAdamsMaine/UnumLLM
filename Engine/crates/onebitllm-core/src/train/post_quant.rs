//! Post-training quantization.
//!
//! Quantize a trained model's full-precision weights to ternary after
//! training is complete. This is an alternative to QAT for simpler
//! quantization workflows.

use crate::nn::Parameter;
use crate::quant::ternary::TernaryWeight;
use crate::quant::{QuantConfig, QuantParams};
use ndarray::{Array, IxDyn};

/// Result of post-training quantization for a single parameter.
#[derive(Debug, Clone)]
pub struct QuantizedParam {
    /// Parameter name.
    pub name: String,
    /// Ternary weights as f32 values (-scale, 0, +scale).
    pub quantized_data: Array<f32, IxDyn>,
    /// The quantization scale(s) used.
    pub quant_params: QuantParams,
    /// Original full-precision data (retained for potential fine-tuning).
    pub original_data: Array<f32, IxDyn>,
}

/// Quantize a single parameter to ternary.
///
/// Returns a `QuantizedParam` with the ternary weights scaled by the
/// computed scale factor.
pub fn quantize_parameter(param: &Parameter, config: &QuantConfig) -> QuantizedParam {
    let shape = param.data.shape().to_vec();
    let flat: Vec<f32> = param.data.iter().copied().collect();
    let quant_params = QuantParams::compute(&flat, &shape, config);

    let quantized: Vec<f32> = flat
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let scale = quant_params.scale_for_index(i);
            let t = TernaryWeight::quantize(v, scale);
            t.to_f32() * scale
        })
        .collect();

    let quantized_data =
        Array::from_shape_vec(IxDyn(&shape), quantized).expect("post-quantization: shape mismatch");

    QuantizedParam {
        name: param.name.clone(),
        quantized_data,
        quant_params,
        original_data: param.data.clone(),
    }
}

/// Quantize all trainable parameters in a list.
pub fn quantize_all_parameters(params: &[&Parameter], config: &QuantConfig) -> Vec<QuantizedParam> {
    params
        .iter()
        .map(|p| quantize_parameter(p, config))
        .collect()
}

/// Apply post-training quantization in-place: replace parameter data
/// with quantized values.
pub fn apply_quantization_inplace(param: &mut Parameter, config: &QuantConfig) {
    let qp = quantize_parameter(param, config);
    param.data = qp.quantized_data;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_parameter() {
        let param = Parameter::new(
            "weight",
            Array::from_shape_vec(IxDyn(&[6]), vec![1.0, -1.0, 0.5, -0.5, 0.1, -0.1]).unwrap(),
        );
        let config = QuantConfig::per_tensor();
        let qp = quantize_parameter(&param, &config);

        assert_eq!(qp.name, "weight");
        assert_eq!(qp.quantized_data.shape(), &[6]);
        assert_eq!(qp.original_data.shape(), &[6]);

        // Quantized values should be ternary * scale
        let scale = qp.quant_params.scales[0];
        for &v in qp.quantized_data.iter() {
            let normalized = if scale > 0.0 { v / scale } else { 0.0 };
            assert!(
                (normalized - 1.0).abs() < 1e-6
                    || (normalized + 1.0).abs() < 1e-6
                    || normalized.abs() < 1e-6,
                "Value {v} (normalized: {normalized}) is not ternary"
            );
        }
    }

    #[test]
    fn test_quantize_all_parameters() {
        let p1 = Parameter::new("w1", Array::from_elem(IxDyn(&[4]), 0.5f32));
        let p2 = Parameter::new("w2", Array::from_elem(IxDyn(&[3]), -0.3f32));
        let config = QuantConfig::per_tensor();

        let quantized = quantize_all_parameters(&[&p1, &p2], &config);
        assert_eq!(quantized.len(), 2);
        assert_eq!(quantized[0].name, "w1");
        assert_eq!(quantized[1].name, "w2");
    }

    #[test]
    fn test_apply_quantization_inplace() {
        let mut param = Parameter::new(
            "weight",
            Array::from_shape_vec(IxDyn(&[4]), vec![0.8, -0.6, 0.2, -0.1]).unwrap(),
        );
        let original = param.data.clone();
        let config = QuantConfig::per_tensor();

        apply_quantization_inplace(&mut param, &config);

        // Data should be different (quantized)
        assert_ne!(param.data, original);
        // Shape should be preserved
        assert_eq!(param.data.shape(), original.shape());
    }

    #[test]
    fn test_quantize_zeros() {
        let param = Parameter::new("zeros", Array::from_elem(IxDyn(&[4]), 0.0f32));
        let config = QuantConfig::per_tensor();
        let qp = quantize_parameter(&param, &config);

        // All zeros should stay zero
        for &v in qp.quantized_data.iter() {
            assert_eq!(v, 0.0);
        }
    }
}
