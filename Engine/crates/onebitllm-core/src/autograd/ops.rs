//! Differentiable operations that record entries on the computation tape.
//!
//! Each operation computes the forward result, records a backward closure
//! on the tape, and returns a new [`Variable`] with the output data.

use std::sync::{Arc, Mutex};
use ndarray::{Array, Axis, IxDyn, Ix2};
use super::tape::Tape;
use super::variable::Variable;
use crate::quant::{QuantConfig, QuantParams};
use crate::quant::ternary::{TernaryWeight, unit_quantize};

// ---------------------------------------------------------------------------
// Helper: get tape from variable, or return detached result
// ---------------------------------------------------------------------------

fn shared_tape(a: &Variable, b: &Variable) -> Option<Arc<Mutex<Tape>>> {
    a.tape.clone().or_else(|| b.tape.clone())
}

// ---------------------------------------------------------------------------
// Matrix multiply: C = A @ B
// ---------------------------------------------------------------------------

/// Matrix multiplication of two 2-D variables.
///
/// Shapes: `a` is (M, K), `b` is (K, N), output is (M, N).
///
/// Backward:
/// - grad_a = grad_output @ b^T
/// - grad_b = a^T @ grad_output
pub fn matmul(a: &Variable, b: &Variable) -> Variable {
    let a_2d = a.data.view().into_dimensionality::<Ix2>().expect("matmul: a must be 2-D");
    let b_2d = b.data.view().into_dimensionality::<Ix2>().expect("matmul: b must be 2-D");
    let out = a_2d.dot(&b_2d).into_dyn();

    if let Some(tape_ref) = shared_tape(a, b) {
        let a_data = a.data.clone();
        let b_data = b.data.clone();
        let a_id = a.id;
        let b_id = b.id;

        let mut tape = tape_ref.lock().unwrap();
        let out_id = tape.alloc_id();
        tape.record(out_id, vec![a_id, b_id], move |grad| {
            let grad_2d = grad.view().into_dimensionality::<Ix2>()
                .expect("matmul backward: grad must be 2-D");
            let a_2d = a_data.view().into_dimensionality::<Ix2>()
                .expect("matmul backward: a must be 2-D");
            let b_2d = b_data.view().into_dimensionality::<Ix2>()
                .expect("matmul backward: b must be 2-D");

            // grad_a = grad @ b^T, shape (M, K)
            let grad_a = grad_2d.dot(&b_2d.t()).into_dyn();
            // grad_b = a^T @ grad, shape (K, N)
            let grad_b = a_2d.t().dot(&grad_2d).into_dyn();

            vec![grad_a, grad_b]
        });

        Variable {
            id: out_id,
            data: out,
            requires_grad: a.requires_grad || b.requires_grad,
            tape: Some(tape_ref.clone()),
        }
    } else {
        Variable::detached(out)
    }
}

// ---------------------------------------------------------------------------
// Element-wise addition: C = A + B
// ---------------------------------------------------------------------------

/// Element-wise addition with broadcasting support.
///
/// Backward: identity gradient to both inputs (with possible broadcast reduction).
pub fn add(a: &Variable, b: &Variable) -> Variable {
    let out = &a.data + &b.data;

    if let Some(tape_ref) = shared_tape(a, b) {
        let a_shape = a.data.shape().to_vec();
        let b_shape = b.data.shape().to_vec();
        let a_id = a.id;
        let b_id = b.id;

        let mut tape = tape_ref.lock().unwrap();
        let out_id = tape.alloc_id();
        tape.record(out_id, vec![a_id, b_id], move |grad| {
            let grad_a = reduce_to_shape(grad, &a_shape);
            let grad_b = reduce_to_shape(grad, &b_shape);
            vec![grad_a, grad_b]
        });

        Variable {
            id: out_id,
            data: out,
            requires_grad: a.requires_grad || b.requires_grad,
            tape: Some(tape_ref.clone()),
        }
    } else {
        Variable::detached(out)
    }
}

// ---------------------------------------------------------------------------
// Element-wise multiplication: C = A * B
// ---------------------------------------------------------------------------

/// Element-wise multiplication.
///
/// Backward: grad_a = grad * b, grad_b = grad * a.
pub fn mul(a: &Variable, b: &Variable) -> Variable {
    let out = &a.data * &b.data;

    if let Some(tape_ref) = shared_tape(a, b) {
        let a_data = a.data.clone();
        let b_data = b.data.clone();
        let a_shape = a.data.shape().to_vec();
        let b_shape = b.data.shape().to_vec();
        let a_id = a.id;
        let b_id = b.id;

        let mut tape = tape_ref.lock().unwrap();
        let out_id = tape.alloc_id();
        tape.record(out_id, vec![a_id, b_id], move |grad| {
            let grad_a = reduce_to_shape(&(grad * &b_data), &a_shape);
            let grad_b = reduce_to_shape(&(grad * &a_data), &b_shape);
            vec![grad_a, grad_b]
        });

        Variable {
            id: out_id,
            data: out,
            requires_grad: a.requires_grad || b.requires_grad,
            tape: Some(tape_ref.clone()),
        }
    } else {
        Variable::detached(out)
    }
}

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

/// Rectified linear unit: max(0, x).
///
/// Backward: grad * (x > 0).
pub fn relu(x: &Variable) -> Variable {
    let out = x.data.mapv(|v| v.max(0.0));

    if let Some(tape_ref) = &x.tape {
        let x_data = x.data.clone();
        let x_id = x.id;

        let mut tape = tape_ref.lock().unwrap();
        let out_id = tape.alloc_id();
        tape.record(out_id, vec![x_id], move |grad| {
            let mask = x_data.mapv(|v| if v > 0.0 { 1.0f32 } else { 0.0 });
            vec![grad * &mask]
        });

        Variable {
            id: out_id,
            data: out,
            requires_grad: x.requires_grad,
            tape: Some(Arc::clone(tape_ref)),
        }
    } else {
        Variable::detached(out)
    }
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

/// Softmax along the given axis.
///
/// Backward: uses the Jacobian-vector product:
///   grad_input_i = sum_j (softmax_j * (delta_ij - softmax_i) * grad_j)
///                = softmax_i * (grad_i - sum_j(softmax_j * grad_j))
pub fn softmax(x: &Variable, axis: usize) -> Variable {
    let out = softmax_forward(&x.data, axis);

    if let Some(tape_ref) = &x.tape {
        let sm = out.clone();
        let x_id = x.id;

        let mut tape = tape_ref.lock().unwrap();
        let out_id = tape.alloc_id();
        tape.record(out_id, vec![x_id], move |grad| {
            // grad_input = sm * (grad - sum(sm * grad, axis=axis, keepdims=true))
            let sg = &sm * grad;
            let sg_sum = sg.sum_axis(Axis(axis)).insert_axis(Axis(axis));
            let result = &sm * &(grad - &sg_sum);
            vec![result]
        });

        Variable {
            id: out_id,
            data: out,
            requires_grad: x.requires_grad,
            tape: Some(Arc::clone(tape_ref)),
        }
    } else {
        Variable::detached(out)
    }
}

fn softmax_forward(data: &Array<f32, IxDyn>, axis: usize) -> Array<f32, IxDyn> {
    let max = data.map_axis(Axis(axis), |lane| {
        lane.fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    }).insert_axis(Axis(axis));
    let shifted = data - &max;
    let exp = shifted.mapv(f32::exp);
    let sum = exp.sum_axis(Axis(axis)).insert_axis(Axis(axis));
    exp / &sum
}

// ---------------------------------------------------------------------------
// Cross-entropy loss
// ---------------------------------------------------------------------------

/// Cross-entropy loss between logits and integer targets.
///
/// `logits` shape: (batch, num_classes)
/// `targets`: 1-D array of class indices (as f32, will be cast to usize).
///
/// Returns a scalar (shape [1]) variable.
///
/// Backward: grad = (softmax(logits) - one_hot(targets)) / batch_size.
pub fn cross_entropy_loss(logits: &Variable, targets: &Array<f32, IxDyn>) -> Variable {
    let batch_size = logits.data.shape()[0];
    let _num_classes = logits.data.shape()[1];

    // Compute softmax of logits
    let sm = softmax_forward(&logits.data, 1);

    // Compute loss: -log(sm[i, targets[i]]) averaged over batch
    let targets_slice = targets.as_slice().expect("targets must be contiguous");
    let mut loss_sum = 0.0f32;
    for (i, &t) in targets_slice.iter().enumerate() {
        let class = t as usize;
        let p = sm[[i, class].as_ref()].max(1e-12);
        loss_sum -= p.ln();
    }
    let loss = loss_sum / batch_size as f32;
    let out = Array::from_elem(IxDyn(&[1]), loss);

    if let Some(tape_ref) = &logits.tape {
        let logits_id = logits.id;
        let targets_owned = targets.clone();

        let mut tape = tape_ref.lock().unwrap();
        let out_id = tape.alloc_id();
        tape.record(out_id, vec![logits_id], move |grad| {
            let grad_scalar = grad.as_slice().unwrap()[0];
            let batch = sm.shape()[0];
            let _n_cls = sm.shape()[1];
            let tgt = targets_owned.as_slice().unwrap();

            let mut grad_logits = sm.clone();
            for i in 0..batch {
                let class = tgt[i] as usize;
                grad_logits[[i, class].as_ref()] -= 1.0;
            }
            // Scale by grad_scalar / batch_size
            let scale = grad_scalar / batch as f32;
            grad_logits.mapv_inplace(|v| v * scale);
            vec![grad_logits]
        });

        Variable {
            id: out_id,
            data: out,
            requires_grad: logits.requires_grad,
            tape: Some(Arc::clone(tape_ref)),
        }
    } else {
        Variable::detached(out)
    }
}

/// Cross-entropy loss between logits and soft target probabilities.
///
/// `logits` shape: (batch, num_classes)
/// `target_probs` shape: (batch, num_classes)
/// `temperature` rescales the student logits before softmax.
pub fn soft_target_cross_entropy_loss(
    logits: &Variable,
    target_probs: &Array<f32, IxDyn>,
    temperature: f32,
) -> Variable {
    let batch_size = logits.data.shape()[0];
    let num_classes = logits.data.shape()[1];
    let safe_temperature = temperature.max(1e-6);

    assert_eq!(
        target_probs.shape(),
        logits.data.shape(),
        "target_probs shape {:?} must match logits shape {:?}",
        target_probs.shape(),
        logits.data.shape()
    );

    let scaled_logits = logits.data.mapv(|value| value / safe_temperature);
    let sm = softmax_forward(&scaled_logits, 1);

    let mut loss_sum = 0.0f32;
    for batch_idx in 0..batch_size {
        for class_idx in 0..num_classes {
            let p = target_probs[[batch_idx, class_idx].as_ref()];
            if p > 0.0 {
                let q = sm[[batch_idx, class_idx].as_ref()].max(1e-12);
                loss_sum -= p * q.ln();
            }
        }
    }
    let loss = loss_sum / batch_size as f32;
    let out = Array::from_elem(IxDyn(&[1]), loss);

    if let Some(tape_ref) = &logits.tape {
        let logits_id = logits.id;
        let targets_owned = target_probs.clone();

        let mut tape = tape_ref.lock().unwrap();
        let out_id = tape.alloc_id();
        tape.record(out_id, vec![logits_id], move |grad| {
            let grad_scalar = grad.as_slice().unwrap()[0];
            let mut grad_logits = sm.clone();
            for batch_idx in 0..batch_size {
                for class_idx in 0..num_classes {
                    grad_logits[[batch_idx, class_idx].as_ref()] -=
                        targets_owned[[batch_idx, class_idx].as_ref()];
                }
            }
            let scale = grad_scalar / (batch_size as f32 * safe_temperature);
            grad_logits.mapv_inplace(|value| value * scale);
            vec![grad_logits]
        });

        Variable {
            id: out_id,
            data: out,
            requires_grad: logits.requires_grad,
            tape: Some(Arc::clone(tape_ref)),
        }
    } else {
        Variable::detached(out)
    }
}

// ---------------------------------------------------------------------------
// Sum
// ---------------------------------------------------------------------------

/// Sum all elements, returning a scalar variable of shape [1].
///
/// Backward: grad = ones_like(input) * upstream_grad.
pub fn sum(x: &Variable) -> Variable {
    let total: f32 = x.data.iter().copied().sum();
    let out = Array::from_elem(IxDyn(&[1]), total);

    if let Some(tape_ref) = &x.tape {
        let x_shape = x.data.raw_dim();
        let x_id = x.id;

        let mut tape = tape_ref.lock().unwrap();
        let out_id = tape.alloc_id();
        tape.record(out_id, vec![x_id], move |grad| {
            let g = grad.as_slice().unwrap()[0];
            vec![Array::from_elem(x_shape, g)]
        });

        Variable {
            id: out_id,
            data: out,
            requires_grad: x.requires_grad,
            tape: Some(Arc::clone(tape_ref)),
        }
    } else {
        Variable::detached(out)
    }
}

// ---------------------------------------------------------------------------
// Mean
// ---------------------------------------------------------------------------

/// Mean of all elements, returning a scalar variable of shape [1].
///
/// Backward: grad = (1/n) * ones_like(input) * upstream_grad.
pub fn mean(x: &Variable) -> Variable {
    let n = x.data.len() as f32;
    let total: f32 = x.data.iter().copied().sum();
    let out = Array::from_elem(IxDyn(&[1]), total / n);

    if let Some(tape_ref) = &x.tape {
        let x_shape = x.data.raw_dim();
        let x_id = x.id;

        let mut tape = tape_ref.lock().unwrap();
        let out_id = tape.alloc_id();
        tape.record(out_id, vec![x_id], move |grad| {
            let g = grad.as_slice().unwrap()[0];
            vec![Array::from_elem(x_shape, g / n)]
        });

        Variable {
            id: out_id,
            data: out,
            requires_grad: x.requires_grad,
            tape: Some(Arc::clone(tape_ref)),
        }
    } else {
        Variable::detached(out)
    }
}

// ---------------------------------------------------------------------------
// Quantize STE
// ---------------------------------------------------------------------------

/// Straight-Through Estimator quantization.
///
/// Forward: quantize weights to ternary using `QuantConfig`, then dequantize back
///          to f32. This simulates the quantization error during training.
///
/// Backward: clipped identity -- gradient is passed through, clamped to [-1, 1].
///           This allows gradient flow through the non-differentiable quantization.
pub fn quantize_ste(w: &Variable, config: &QuantConfig) -> Variable {
    let shape = w.data.shape().to_vec();
    let flat: Vec<f32> = w.data.iter().copied().collect();

    // Compute quantization parameters
    let params = QuantParams::compute(&flat, &shape, config);

    // Quantize to ternary and immediately dequantize
    let dequant: Vec<f32> = flat
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let scale = params.scale_for_index(i);
            let t = TernaryWeight::quantize(v, scale);
            t.to_f32() * scale
        })
        .collect();

    let out = Array::from_shape_vec(IxDyn(&shape), dequant)
        .expect("quantize_ste: shape mismatch in output");

    if let Some(tape_ref) = &w.tape {
        let w_id = w.id;

        let mut tape = tape_ref.lock().unwrap();
        let out_id = tape.alloc_id();
        tape.record(out_id, vec![w_id], move |grad| {
            // STE: clipped identity gradient
            vec![grad.mapv(|g| g.clamp(-1.0, 1.0))]
        });

        Variable {
            id: out_id,
            data: out,
            requires_grad: w.requires_grad,
            tape: Some(Arc::clone(tape_ref)),
        }
    } else {
        Variable::detached(out)
    }
}

/// Straight-Through Estimator quantization to unit ternary {-1, 0, +1}.
///
/// Forward: quantize weights to exact unit ternary values using a fixed
/// threshold. Backward: clipped identity gradient.
pub fn quantize_unit_ste(w: &Variable, threshold: f32) -> Variable {
    let shape = w.data.shape().to_vec();
    let flat: Vec<f32> = w.data.iter().copied().collect();
    let ternary = unit_quantize(&flat, threshold);
    let out = Array::from_shape_vec(
        IxDyn(&shape),
        ternary.into_iter().map(TernaryWeight::to_f32).collect(),
    )
    .expect("quantize_unit_ste: shape mismatch in output");

    if let Some(tape_ref) = &w.tape {
        let w_id = w.id;

        let mut tape = tape_ref.lock().unwrap();
        let out_id = tape.alloc_id();
        tape.record(out_id, vec![w_id], move |grad| {
            vec![grad.mapv(|g| g.clamp(-1.0, 1.0))]
        });

        Variable {
            id: out_id,
            data: out,
            requires_grad: w.requires_grad,
            tape: Some(Arc::clone(tape_ref)),
        }
    } else {
        Variable::detached(out)
    }
}

// ---------------------------------------------------------------------------
// RMS Norm
// ---------------------------------------------------------------------------

/// Root Mean Square Layer Normalization.
///
/// `x` shape: (..., dim), `weight` shape: (dim,).
/// output = x / rms(x) * weight, where rms(x) = sqrt(mean(x^2) + eps).
///
/// Backward is approximated: we compute the gradient through the normalization
/// and weight multiplication.
pub fn rms_norm(x: &Variable, weight: &Variable, eps: f32) -> Variable {
    let x_data = &x.data;
    let w_data = &weight.data;

    // Compute RMS along the last axis
    let ndim = x_data.ndim();
    let last_axis = ndim - 1;
    let x_sq = x_data.mapv(|v| v * v);
    let mean_sq = x_sq.mean_axis(Axis(last_axis)).unwrap().insert_axis(Axis(last_axis));
    let rms = mean_sq.mapv(|v| (v + eps).sqrt());
    let x_norm = x_data / &rms;
    let out = &x_norm * w_data;

    if let Some(tape_ref) = shared_tape(x, weight) {
        let x_id = x.id;
        let w_id = weight.id;
        let x_saved = x.data.clone();
        let w_saved = weight.data.clone();
        let rms_saved = rms.clone();
        let x_norm_saved = x_norm.clone();

        let mut tape = tape_ref.lock().unwrap();
        let out_id = tape.alloc_id();
        tape.record(out_id, vec![x_id, w_id], move |grad| {
            let _dim = x_saved.shape()[x_saved.ndim() - 1] as f32;

            // grad_x_norm = grad * weight
            let grad_x_norm = grad * &w_saved;

            // Gradient through the normalization: simplified form
            // d(x/rms)/dx = (1/rms) * (I - (1/dim) * x_norm * x_norm^T) applied per-row
            // Simplified: grad_x = (1/rms) * (grad_x_norm - x_norm * mean(grad_x_norm * x_norm, last_axis))
            let last = x_saved.ndim() - 1;
            let inner = (&grad_x_norm * &x_norm_saved)
                .mean_axis(Axis(last))
                .unwrap()
                .insert_axis(Axis(last));
            let grad_x = (&grad_x_norm - &x_norm_saved * &inner) / &rms_saved;

            // grad_weight = sum over batch of grad * x_norm
            let grad_w_full = grad * &x_norm_saved;
            // Sum over all axes except the last
            let mut grad_w = grad_w_full;
            for ax in (0..last).rev() {
                grad_w = grad_w.sum_axis(Axis(ax));
            }

            vec![grad_x, grad_w]
        });

        Variable {
            id: out_id,
            data: out,
            requires_grad: x.requires_grad || weight.requires_grad,
            tape: Some(tape_ref.clone()),
        }
    } else {
        Variable::detached(out)
    }
}

// ---------------------------------------------------------------------------
// Reshape
// ---------------------------------------------------------------------------

/// Reshape a variable to a new shape. The total number of elements must match.
///
/// Backward: reshapes the gradient back to the original shape.
pub fn reshape(x: &Variable, new_shape: &[usize]) -> Variable {
    let out = x.data.clone()
        .into_shape_with_order(IxDyn(new_shape))
        .expect("reshape: incompatible shape");

    if let Some(tape_ref) = &x.tape {
        let old_shape: Vec<usize> = x.data.shape().to_vec();
        let x_id = x.id;

        let mut tape = tape_ref.lock().unwrap();
        let out_id = tape.alloc_id();
        tape.record(out_id, vec![x_id], move |grad| {
            let reshaped = grad.clone()
                .into_shape_with_order(IxDyn(&old_shape))
                .expect("reshape backward: incompatible shape");
            vec![reshaped]
        });

        Variable {
            id: out_id,
            data: out,
            requires_grad: x.requires_grad,
            tape: Some(Arc::clone(tape_ref)),
        }
    } else {
        Variable::detached(out)
    }
}

// ---------------------------------------------------------------------------
// Broadcast reduction helper
// ---------------------------------------------------------------------------

/// Reduce a gradient array to the target shape by summing along broadcast
/// dimensions.
fn reduce_to_shape(grad: &Array<f32, IxDyn>, target_shape: &[usize]) -> Array<f32, IxDyn> {
    let grad_shape = grad.shape();
    if grad_shape == target_shape {
        return grad.clone();
    }

    let mut result = grad.clone();

    // Handle case where target has fewer dims (was broadcast from fewer dims)
    while result.ndim() > target_shape.len() {
        result = result.sum_axis(Axis(0));
    }

    // Sum along axes where target_shape has size 1
    for (i, (&gs, &ts)) in result.shape().to_vec().iter().zip(target_shape.iter()).enumerate().rev() {
        if ts == 1 && gs != 1 {
            result = result.sum_axis(Axis(i)).insert_axis(Axis(i));
        }
    }

    result
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    /// Helper: create a tracked variable from a 1-D vec.
    fn var1d(tape: &Arc<Mutex<Tape>>, data: &[f32]) -> Variable {
        Variable::new(
            Array::from_shape_vec(IxDyn(&[data.len()]), data.to_vec()).unwrap(),
            true,
            tape,
        )
    }

    /// Helper: create a tracked 2-D variable.
    fn var2d(tape: &Arc<Mutex<Tape>>, rows: usize, cols: usize, data: Vec<f32>) -> Variable {
        Variable::new(
            Array::from_shape_vec(IxDyn(&[rows, cols]), data).unwrap(),
            true,
            tape,
        )
    }

    // --- x^2 gradient test ---

    #[test]
    fn test_gradient_x_squared() {
        // f(x) = sum(x * x), df/dx = 2 * x
        let tape = Tape::new();
        let x = var1d(&tape, &[1.0, 2.0, 3.0]);
        let x2 = mul(&x, &x); // x * x
        let loss = sum(&x2);

        let grads = loss.backward().unwrap();
        let grad_x = grads.get(&x.id).unwrap();

        // df/dx_i = 2 * x_i
        let expected = [2.0, 4.0, 6.0];
        for (g, e) in grad_x.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-5, "Expected {e}, got {g}");
        }
    }

    // --- matmul gradient test ---

    #[test]
    fn test_matmul_gradient() {
        let tape = Tape::new();
        // A: 2x3, B: 3x2
        let a = var2d(&tape, 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = var2d(&tape, 3, 2, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);

        let c = matmul(&a, &b); // (2, 2)
        let loss = sum(&c); // scalar

        let grads = loss.backward().unwrap();
        let grad_a = grads.get(&a.id).unwrap();
        let grad_b = grads.get(&b.id).unwrap();

        // Verify with finite differences
        let eps = 1e-3;
        for idx in 0..6 {
            let mut a_plus = a.data.clone();
            let mut a_minus = a.data.clone();
            a_plus.as_slice_mut().unwrap()[idx] += eps;
            a_minus.as_slice_mut().unwrap()[idx] -= eps;

            let a_p_2d = a_plus.view().into_dimensionality::<Ix2>().unwrap();
            let a_m_2d = a_minus.view().into_dimensionality::<Ix2>().unwrap();
            let b_2d = b.data.view().into_dimensionality::<Ix2>().unwrap();

            let f_plus: f32 = a_p_2d.dot(&b_2d).iter().sum();
            let f_minus: f32 = a_m_2d.dot(&b_2d).iter().sum();
            let fd_grad = (f_plus - f_minus) / (2.0 * eps);
            let ad_grad = grad_a.as_slice().unwrap()[idx];

            assert!(
                (fd_grad - ad_grad).abs() < 1e-2,
                "matmul grad_a[{idx}]: finite diff={fd_grad}, autograd={ad_grad}"
            );
        }

        for idx in 0..6 {
            let mut b_plus = b.data.clone();
            let mut b_minus = b.data.clone();
            b_plus.as_slice_mut().unwrap()[idx] += eps;
            b_minus.as_slice_mut().unwrap()[idx] -= eps;

            let a_2d = a.data.view().into_dimensionality::<Ix2>().unwrap();
            let b_p_2d = b_plus.view().into_dimensionality::<Ix2>().unwrap();
            let b_m_2d = b_minus.view().into_dimensionality::<Ix2>().unwrap();

            let f_plus: f32 = a_2d.dot(&b_p_2d).iter().sum();
            let f_minus: f32 = a_2d.dot(&b_m_2d).iter().sum();
            let fd_grad = (f_plus - f_minus) / (2.0 * eps);
            let ad_grad = grad_b.as_slice().unwrap()[idx];

            assert!(
                (fd_grad - ad_grad).abs() < 1e-2,
                "matmul grad_b[{idx}]: finite diff={fd_grad}, autograd={ad_grad}"
            );
        }
    }

    // --- add test ---

    #[test]
    fn test_add_gradient() {
        let tape = Tape::new();
        let a = var1d(&tape, &[1.0, 2.0, 3.0]);
        let b = var1d(&tape, &[4.0, 5.0, 6.0]);
        let c = add(&a, &b);
        let loss = sum(&c);

        let grads = loss.backward().unwrap();
        let grad_a = grads.get(&a.id).unwrap();
        let grad_b = grads.get(&b.id).unwrap();

        // d(sum(a+b))/da = [1,1,1], d(sum(a+b))/db = [1,1,1]
        for g in grad_a.iter() {
            assert!((g - 1.0).abs() < 1e-6);
        }
        for g in grad_b.iter() {
            assert!((g - 1.0).abs() < 1e-6);
        }
    }

    // --- mul test ---

    #[test]
    fn test_mul_gradient() {
        let tape = Tape::new();
        let a = var1d(&tape, &[2.0, 3.0]);
        let b = var1d(&tape, &[4.0, 5.0]);
        let c = mul(&a, &b); // [8, 15]
        let loss = sum(&c);

        let grads = loss.backward().unwrap();
        let grad_a = grads.get(&a.id).unwrap();
        let grad_b = grads.get(&b.id).unwrap();

        // d(sum(a*b))/da_i = b_i
        assert!((grad_a.as_slice().unwrap()[0] - 4.0).abs() < 1e-6);
        assert!((grad_a.as_slice().unwrap()[1] - 5.0).abs() < 1e-6);
        // d(sum(a*b))/db_i = a_i
        assert!((grad_b.as_slice().unwrap()[0] - 2.0).abs() < 1e-6);
        assert!((grad_b.as_slice().unwrap()[1] - 3.0).abs() < 1e-6);
    }

    // --- relu test ---

    #[test]
    fn test_relu_gradient() {
        let tape = Tape::new();
        let x = var1d(&tape, &[-1.0, 0.0, 1.0, 2.0]);
        let y = relu(&x);
        let loss = sum(&y);

        let grads = loss.backward().unwrap();
        let grad_x = grads.get(&x.id).unwrap();

        // relu grad: 0 for x<=0, 1 for x>0
        let expected = [0.0, 0.0, 1.0, 1.0];
        for (g, e) in grad_x.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6);
        }
    }

    // --- softmax test ---

    #[test]
    fn test_softmax_forward() {
        let tape = Tape::new();
        let x = Variable::new(
            Array::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap(),
            true,
            &tape,
        );
        let sm = softmax(&x, 1);

        // Softmax should sum to 1
        let total: f32 = sm.data.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);

        // Values should be monotonically increasing
        let s = sm.data.as_slice().unwrap();
        assert!(s[0] < s[1] && s[1] < s[2]);
    }

    // --- cross_entropy test ---

    #[test]
    fn test_cross_entropy_gradient() {
        let tape = Tape::new();
        let logits = Variable::new(
            Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 1.0, 0.5, 0.1]).unwrap(),
            true,
            &tape,
        );
        let targets = Array::from_shape_vec(IxDyn(&[2]), vec![2.0, 0.0]).unwrap();

        let loss = cross_entropy_loss(&logits, &targets);
        assert_eq!(loss.data.shape(), &[1]);

        let grads = loss.backward().unwrap();
        let grad_logits = grads.get(&logits.id).unwrap();
        assert_eq!(grad_logits.shape(), &[2, 3]);

        // Gradient rows should sum to ~0 (softmax - one_hot sums to 0)
        let g = grad_logits.as_slice().unwrap();
        let row0_sum: f32 = g[0..3].iter().sum();
        let row1_sum: f32 = g[3..6].iter().sum();
        assert!(row0_sum.abs() < 1e-5);
        assert!(row1_sum.abs() < 1e-5);
    }

    #[test]
    fn test_soft_target_cross_entropy_gradient() {
        let tape = Tape::new();
        let logits = Variable::new(
            Array::from_shape_vec(IxDyn(&[1, 3]), vec![3.0, 1.0, -1.0]).unwrap(),
            true,
            &tape,
        );
        let targets =
            Array::from_shape_vec(IxDyn(&[1, 3]), vec![0.9, 0.1, 0.0]).unwrap();

        let loss = soft_target_cross_entropy_loss(&logits, &targets, 1.0);
        let grads = loss.backward().unwrap();
        let grad_logits = grads.get(&logits.id).unwrap();

        assert!(loss.data[[0]] > 0.0);
        assert!(grad_logits[[0, 0]] < 0.0);
        assert!(grad_logits[[0, 2]] > 0.0);
    }

    // --- sum test ---

    #[test]
    fn test_sum_gradient() {
        let tape = Tape::new();
        let x = var1d(&tape, &[1.0, 2.0, 3.0]);
        let s = sum(&x);

        assert_eq!(s.data.as_slice().unwrap(), &[6.0]);

        let grads = s.backward().unwrap();
        let grad_x = grads.get(&x.id).unwrap();
        for g in grad_x.iter() {
            assert!((g - 1.0).abs() < 1e-6);
        }
    }

    // --- mean test ---

    #[test]
    fn test_mean_gradient() {
        let tape = Tape::new();
        let x = var1d(&tape, &[2.0, 4.0, 6.0]);
        let m = mean(&x);

        assert!((m.data.as_slice().unwrap()[0] - 4.0).abs() < 1e-6);

        let grads = m.backward().unwrap();
        let grad_x = grads.get(&x.id).unwrap();
        for g in grad_x.iter() {
            assert!((g - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    // --- quantize STE test ---

    #[test]
    fn test_quantize_ste_gradient_flows() {
        // STE should pass gradient through (clipped to [-1, 1])
        let tape = Tape::new();
        let w = var1d(&tape, &[0.8, -0.9, 0.1, 0.5]);
        let config = QuantConfig::per_tensor();
        let q = quantize_ste(&w, &config);
        let loss = sum(&q);

        let grads = loss.backward().unwrap();
        let grad_w = grads.get(&w.id).unwrap();

        // All gradients should be clipped to [-1, 1]
        for g in grad_w.iter() {
            assert!(*g >= -1.0 && *g <= 1.0,
                "STE gradient {g} out of [-1, 1] range");
        }
        // Since upstream grad is all 1s and we clip, grad should be 1.0
        for g in grad_w.iter() {
            assert!((g - 1.0).abs() < 1e-6,
                "STE gradient should be 1.0, got {g}");
        }
    }

    #[test]
    fn test_quantize_ste_clips_large_gradient() {
        // Build a chain where upstream gradient is large
        let tape = Tape::new();
        let w = var1d(&tape, &[0.5]);
        let config = QuantConfig::per_tensor();
        let q = quantize_ste(&w, &config);

        // Manually scale output to make gradient large: loss = 10 * q
        let scale = Variable::new(
            Array::from_elem(IxDyn(&[1]), 10.0f32),
            false,
            &tape,
        );
        let scaled = mul(&q, &scale);
        let loss = sum(&scaled);

        let grads = loss.backward().unwrap();
        let grad_w = grads.get(&w.id).unwrap();

        // Upstream grad to STE is 10.0, should be clipped to 1.0
        for g in grad_w.iter() {
            assert!((g - 1.0).abs() < 1e-6,
                "Expected clipped grad 1.0, got {g}");
        }
    }

    #[test]
    fn test_quantize_unit_ste_forward_outputs_exact_ternary() {
        let tape = Tape::new();
        let w = var1d(&tape, &[0.8, -0.8, 0.2, -0.2]);
        let q = quantize_unit_ste(&w, 0.5);
        assert_eq!(q.data.as_slice().unwrap(), &[1.0, -1.0, 0.0, 0.0]);
    }

    // --- rms_norm test ---

    #[test]
    fn test_rms_norm_forward() {
        let tape = Tape::new();
        let x = Variable::new(
            Array::from_shape_vec(IxDyn(&[2, 4]), vec![
                1.0, 2.0, 3.0, 4.0,
                -1.0, -2.0, -3.0, -4.0,
            ]).unwrap(),
            true,
            &tape,
        );
        let w = Variable::new(
            Array::from_elem(IxDyn(&[4]), 1.0f32),
            true,
            &tape,
        );
        let out = rms_norm(&x, &w, 1e-6);
        assert_eq!(out.data.shape(), &[2, 4]);

        // With weight=1, output should have RMS ~1 along last axis
        let row0: Vec<f32> = (0..4).map(|j| out.data[[0, j].as_ref()]).collect();
        let rms: f32 = (row0.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 0.1, "RMS should be ~1.0, got {rms}");
    }

    #[test]
    fn test_rms_norm_gradient() {
        let tape = Tape::new();
        let x = Variable::new(
            Array::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap(),
            true,
            &tape,
        );
        let w = Variable::new(
            Array::from_elem(IxDyn(&[3]), 1.0f32),
            true,
            &tape,
        );
        let out = rms_norm(&x, &w, 1e-6);
        let loss = sum(&out);
        let grads = loss.backward().unwrap();

        // Just check gradients exist and have right shape
        let grad_x = grads.get(&x.id).unwrap();
        let grad_w = grads.get(&w.id).unwrap();
        assert_eq!(grad_x.shape(), &[1, 3]);
        assert_eq!(grad_w.shape(), &[3]);
    }

    // --- reshape test ---

    #[test]
    fn test_reshape_gradient() {
        let tape = Tape::new();
        let x = var2d(&tape, 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = reshape(&x, &[3, 2]);
        assert_eq!(y.data.shape(), &[3, 2]);

        let loss = sum(&y);
        let grads = loss.backward().unwrap();
        let grad_x = grads.get(&x.id).unwrap();

        // Gradient should be all 1s with original shape
        assert_eq!(grad_x.shape(), &[2, 3]);
        for g in grad_x.iter() {
            assert!((g - 1.0).abs() < 1e-6);
        }
    }

    // --- chain test ---

    #[test]
    fn test_chain_matmul_add_relu() {
        let tape = Tape::new();
        // x: (1, 2), w: (2, 2), b: (1, 2)
        let x = var2d(&tape, 1, 2, vec![1.0, -1.0]);
        let w = var2d(&tape, 2, 2, vec![0.5, 0.3, -0.2, 0.4]);
        let b = var2d(&tape, 1, 2, vec![0.1, -0.1]);

        let xw = matmul(&x, &w);
        let xwb = add(&xw, &b);
        let out = relu(&xwb);
        let loss = sum(&out);

        let grads = loss.backward().unwrap();
        assert!(grads.contains_key(&x.id));
        assert!(grads.contains_key(&w.id));
        assert!(grads.contains_key(&b.id));

        // Verify shape consistency
        assert_eq!(grads[&x.id].shape(), &[1, 2]);
        assert_eq!(grads[&w.id].shape(), &[2, 2]);
        assert_eq!(grads[&b.id].shape(), &[1, 2]);
    }

    // --- broadcast add test ---

    #[test]
    fn test_add_broadcast_gradient() {
        let tape = Tape::new();
        // a: (2, 3), b: (1, 3) -- b is broadcast along axis 0
        let a = var2d(&tape, 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Variable::new(
            Array::from_shape_vec(IxDyn(&[1, 3]), vec![0.1, 0.2, 0.3]).unwrap(),
            true,
            &tape,
        );
        let c = add(&a, &b);
        let loss = sum(&c);
        let grads = loss.backward().unwrap();

        // grad_a: same shape (2,3), all 1s
        let grad_a = &grads[&a.id];
        assert_eq!(grad_a.shape(), &[2, 3]);

        // grad_b: reduced to (1,3), each = 2 (summed over broadcast dim)
        let grad_b = &grads[&b.id];
        assert_eq!(grad_b.shape(), &[1, 3]);
        for g in grad_b.iter() {
            assert!((g - 2.0).abs() < 1e-6);
        }
    }
}
