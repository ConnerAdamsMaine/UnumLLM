use ndarray::{Array, IxDyn};

/// Supported activation functions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationFn {
    ReLU,
    GELU,
    SiLU, // aka Swish
    Mish,
    /// SwiGLU is handled specially in MLP (gate * silu(x)); this variant is just SiLU.
    SwiGLU,
}

impl ActivationFn {
    /// Apply activation element-wise to an array.
    pub fn apply(&self, x: &Array<f32, IxDyn>) -> Array<f32, IxDyn> {
        match self {
            ActivationFn::ReLU => x.mapv(relu),
            ActivationFn::GELU => x.mapv(gelu),
            ActivationFn::SiLU | ActivationFn::SwiGLU => x.mapv(silu),
            ActivationFn::Mish => x.mapv(mish),
        }
    }

    /// Apply activation derivative element-wise (for backpropagation).
    pub fn derivative(&self, x: &Array<f32, IxDyn>) -> Array<f32, IxDyn> {
        match self {
            ActivationFn::ReLU => x.mapv(relu_derivative),
            ActivationFn::GELU => x.mapv(gelu_derivative),
            ActivationFn::SiLU | ActivationFn::SwiGLU => x.mapv(silu_derivative),
            ActivationFn::Mish => x.mapv(mish_derivative),
        }
    }
}

#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn gelu(x: f32) -> f32 {
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

#[inline]
fn gelu_derivative(x: f32) -> f32 {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    let inner = c * (x + 0.044715 * x * x * x);
    let tanh_val = inner.tanh();
    let sech2 = 1.0 - tanh_val * tanh_val;
    let inner_deriv = c * (1.0 + 3.0 * 0.044715 * x * x);
    0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * inner_deriv
}

#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
fn silu_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    s + x * s * (1.0 - s)
}

#[inline]
fn mish(x: f32) -> f32 {
    x * ((1.0 + x.exp()).ln()).tanh()
}

#[inline]
fn mish_derivative(x: f32) -> f32 {
    let sp = (1.0 + x.exp()).ln(); // softplus
    let tanh_sp = sp.tanh();
    let sech2_sp = 1.0 - tanh_sp * tanh_sp;
    let sigmoid_x = sigmoid(x);
    tanh_sp + x * sech2_sp * sigmoid_x
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_relu() {
        let x = array![-1.0f32, 0.0, 1.0, 2.0].into_dyn();
        let result = ActivationFn::ReLU.apply(&x);
        assert_eq!(result, array![0.0f32, 0.0, 1.0, 2.0].into_dyn());
    }

    #[test]
    fn test_relu_derivative() {
        let x = array![-1.0f32, 0.0, 1.0].into_dyn();
        let result = ActivationFn::ReLU.derivative(&x);
        assert_eq!(result, array![0.0f32, 0.0, 1.0].into_dyn());
    }

    #[test]
    fn test_gelu_at_zero() {
        let x = array![0.0f32].into_dyn();
        let result = ActivationFn::GELU.apply(&x);
        assert!((result[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_positive() {
        let x = array![1.0f32].into_dyn();
        let result = ActivationFn::GELU.apply(&x);
        // GELU(1) ≈ 0.8413
        assert!((result[0] - 0.8413).abs() < 0.01);
    }

    #[test]
    fn test_silu_at_zero() {
        let x = array![0.0f32].into_dyn();
        let result = ActivationFn::SiLU.apply(&x);
        assert!((result[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_silu_positive() {
        let x = array![1.0f32].into_dyn();
        let result = ActivationFn::SiLU.apply(&x);
        // SiLU(1) = 1 * sigmoid(1) ≈ 0.7311
        assert!((result[0] - 0.7311).abs() < 0.01);
    }

    #[test]
    fn test_mish_at_zero() {
        let x = array![0.0f32].into_dyn();
        let result = ActivationFn::Mish.apply(&x);
        assert!((result[0] - 0.0).abs() < 1e-6);
    }
}
