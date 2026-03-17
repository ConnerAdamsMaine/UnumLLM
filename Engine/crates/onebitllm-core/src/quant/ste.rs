/// Straight-Through Estimator (STE) gradient functions.
///
/// During the forward pass, weights are quantized to ternary {-1, 0, +1}.
/// During the backward pass, the gradient must flow through this discrete
/// quantization step. STE variants approximate this gradient.

/// Clipped STE: pass gradient through, clipped to [-1, 1].
///
/// This is the standard STE used in BitNet b1.58. It prevents gradient
/// explosion by clamping large gradients.
#[inline]
pub fn ste_clip_grad(grad: f32) -> f32 {
    grad.clamp(-1.0, 1.0)
}

/// Identity STE: pass gradient through unchanged.
///
/// Simplest STE variant. Can lead to large gradients but is useful
/// for debugging and comparison.
#[inline]
pub fn ste_identity_grad(grad: f32) -> f32 {
    grad
}

/// Polynomial STE approximation.
///
/// `grad * max(0, 1 - w_fp^2)` where `w_fp` is the full-precision weight.
/// This naturally attenuates gradients for weights far from zero and provides
/// a smooth approximation to the quantization gradient.
#[inline]
pub fn ste_polynomial_grad(grad: f32, w_fp: f32) -> f32 {
    let factor = (1.0 - w_fp * w_fp).max(0.0);
    grad * factor
}

/// Apply clipped STE to an entire slice of gradients in-place.
pub fn ste_clip_grad_slice(grads: &mut [f32]) {
    for g in grads.iter_mut() {
        *g = ste_clip_grad(*g);
    }
}

/// Apply polynomial STE to slices of gradients and weights in-place.
pub fn ste_polynomial_grad_slice(grads: &mut [f32], weights: &[f32]) {
    for (g, w) in grads.iter_mut().zip(weights.iter()) {
        *g = ste_polynomial_grad(*g, *w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_ste() {
        assert_eq!(ste_clip_grad(0.5), 0.5);
        assert_eq!(ste_clip_grad(-0.5), -0.5);
        assert_eq!(ste_clip_grad(2.0), 1.0);
        assert_eq!(ste_clip_grad(-2.0), -1.0);
        assert_eq!(ste_clip_grad(0.0), 0.0);
        assert_eq!(ste_clip_grad(1.0), 1.0);
        assert_eq!(ste_clip_grad(-1.0), -1.0);
    }

    #[test]
    fn test_identity_ste() {
        assert_eq!(ste_identity_grad(0.5), 0.5);
        assert_eq!(ste_identity_grad(5.0), 5.0);
        assert_eq!(ste_identity_grad(-5.0), -5.0);
    }

    #[test]
    fn test_polynomial_ste() {
        // w_fp = 0 -> factor = 1, grad passes through
        assert_eq!(ste_polynomial_grad(0.5, 0.0), 0.5);

        // w_fp = 1 -> factor = 0, grad is zeroed
        assert_eq!(ste_polynomial_grad(0.5, 1.0), 0.0);

        // w_fp = -1 -> factor = 0, grad is zeroed
        assert_eq!(ste_polynomial_grad(0.5, -1.0), 0.0);

        // w_fp = 0.5 -> factor = 1 - 0.25 = 0.75
        let result = ste_polynomial_grad(1.0, 0.5);
        assert!((result - 0.75).abs() < 1e-6);

        // w_fp = 2.0 -> factor = max(0, 1-4) = 0
        assert_eq!(ste_polynomial_grad(1.0, 2.0), 0.0);
    }

    #[test]
    fn test_clip_grad_slice() {
        let mut grads = vec![0.5, -0.5, 2.0, -2.0, 0.0];
        ste_clip_grad_slice(&mut grads);
        assert_eq!(grads, vec![0.5, -0.5, 1.0, -1.0, 0.0]);
    }

    #[test]
    fn test_polynomial_grad_slice() {
        let mut grads = vec![1.0, 1.0, 1.0];
        let weights = vec![0.0, 0.5, 1.0];
        ste_polynomial_grad_slice(&mut grads, &weights);
        assert!((grads[0] - 1.0).abs() < 1e-6);
        assert!((grads[1] - 0.75).abs() < 1e-6);
        assert!((grads[2] - 0.0).abs() < 1e-6);
    }
}
