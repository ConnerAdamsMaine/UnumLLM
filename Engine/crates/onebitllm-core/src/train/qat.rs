//! Quantization-Aware Training (QAT) utilities.
//!
//! During QAT, forward passes quantize weights to ternary using the
//! Straight-Through Estimator for gradient flow. These functions
//! manage enabling and freezing QAT mode on model parameters.

use crate::autograd::ops;
use crate::autograd::Variable;
use crate::nn::Parameter;
use crate::quant::QuantConfig;
use ndarray::{Array, IxDyn};

/// Apply QAT quantization to a weight variable.
///
/// This wraps the weight through `quantize_ste`: the forward pass sees
/// ternary-quantized weights, but gradients flow through via STE.
pub fn qat_quantize_weight(w: &Variable, config: &QuantConfig) -> Variable {
    ops::quantize_ste(w, config)
}

/// Freeze all parameters in a list (set requires_grad = false).
///
/// Used after QAT to lock parameters before inference.
pub fn freeze_parameters(params: &mut [&mut Parameter]) {
    for param in params.iter_mut() {
        param.requires_grad = false;
    }
}

/// Unfreeze all parameters in a list (set requires_grad = true).
///
/// Used to re-enable training after a freeze.
pub fn unfreeze_parameters(params: &mut [&mut Parameter]) {
    for param in params.iter_mut() {
        param.requires_grad = true;
    }
}

/// Check if a parameter's weights are approximately ternary
/// (all values close to -scale, 0, or +scale).
pub fn is_approximately_ternary(data: &Array<f32, IxDyn>, tolerance: f32) -> bool {
    if data.is_empty() {
        return true;
    }

    // Find the scale (largest absolute value that appears frequently)
    let abs_vals: Vec<f32> = data.iter().map(|v| v.abs()).collect();
    let max_abs = abs_vals.iter().cloned().fold(0.0f32, f32::max);

    if max_abs < tolerance {
        // All zeros
        return true;
    }

    // Check each value is close to 0, +max_abs, or -max_abs
    data.iter().all(|&v| {
        let abs_v = v.abs();
        abs_v < tolerance || (abs_v - max_abs).abs() < tolerance
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Tape;

    #[test]
    fn test_qat_quantize_weight() {
        let tape = Tape::new();
        let w = Variable::new(
            Array::from_shape_vec(IxDyn(&[4]), vec![0.8, -0.9, 0.1, 0.5]).unwrap(),
            true,
            &tape,
        );
        let config = QuantConfig::per_tensor();
        let q = qat_quantize_weight(&w, &config);

        // Output should be quantized (ternary * scale)
        assert_eq!(q.data.shape(), w.data.shape());
        assert!(q.tape.is_some());
    }

    #[test]
    fn test_freeze_unfreeze() {
        let mut p1 = Parameter::new("a", Array::from_elem(IxDyn(&[2]), 1.0f32));
        let mut p2 = Parameter::new("b", Array::from_elem(IxDyn(&[3]), 2.0f32));

        assert!(p1.requires_grad);
        assert!(p2.requires_grad);

        freeze_parameters(&mut [&mut p1, &mut p2]);
        assert!(!p1.requires_grad);
        assert!(!p2.requires_grad);

        unfreeze_parameters(&mut [&mut p1, &mut p2]);
        assert!(p1.requires_grad);
        assert!(p2.requires_grad);
    }

    #[test]
    fn test_is_approximately_ternary() {
        // Perfectly ternary (with scale 0.5)
        let data =
            Array::from_shape_vec(IxDyn(&[6]), vec![0.5, -0.5, 0.0, 0.5, 0.0, -0.5]).unwrap();
        assert!(is_approximately_ternary(&data, 0.01));

        // Not ternary
        let data = Array::from_shape_vec(IxDyn(&[3]), vec![0.3, 0.7, -0.5]).unwrap();
        assert!(!is_approximately_ternary(&data, 0.01));

        // Empty is ternary
        let data = Array::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();
        assert!(is_approximately_ternary(&data, 0.01));
    }

    #[test]
    fn test_qat_gradient_flow() {
        use crate::autograd::ops;

        let tape = Tape::new();
        let w = Variable::new(
            Array::from_shape_vec(IxDyn(&[3]), vec![0.8, -0.9, 0.1]).unwrap(),
            true,
            &tape,
        );
        let config = QuantConfig::per_tensor();
        let q = qat_quantize_weight(&w, &config);
        let loss = ops::sum(&q);

        let grads = loss.backward().unwrap();
        let grad_w = grads.get(&w.id).unwrap();

        // STE should pass gradient through (clipped to [-1,1])
        assert_eq!(grad_w.shape(), &[3]);
        for g in grad_w.iter() {
            assert!(*g >= -1.0 && *g <= 1.0);
        }
    }
}
