use std::collections::HashMap;
use ndarray::{Array, IxDyn};
use crate::nn::Parameter;
use crate::autograd::VarId;
use super::traits::Optimizer;

/// Stochastic Gradient Descent with optional momentum.
///
/// When momentum > 0, uses the classical momentum formulation:
///   v = momentum * v + grad
///   param -= lr * v
pub struct Sgd {
    lr: f32,
    momentum: f32,
    /// Velocity buffers (per VarId), only populated when momentum > 0.
    velocity: HashMap<VarId, Array<f32, IxDyn>>,
}

impl Sgd {
    /// Create a new SGD optimizer.
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            velocity: HashMap::new(),
        }
    }

    /// Create SGD without momentum.
    pub fn vanilla(lr: f32) -> Self {
        Self::new(lr, 0.0)
    }
}

impl Optimizer for Sgd {
    fn step(
        &mut self,
        params: &mut [&mut Parameter],
        grads: &HashMap<VarId, Array<f32, IxDyn>>,
        param_ids: &[VarId],
    ) -> crate::Result<()> {
        if params.len() != param_ids.len() {
            return Err(crate::error::OneBitError::Training(
                format!(
                    "params length ({}) != param_ids length ({})",
                    params.len(),
                    param_ids.len()
                ),
            ));
        }

        for (param, &var_id) in params.iter_mut().zip(param_ids.iter()) {
            if !param.requires_grad {
                continue;
            }

            let grad = match grads.get(&var_id) {
                Some(g) => g,
                None => continue,
            };

            if self.momentum > 0.0 {
                let vel = self.velocity
                    .entry(var_id)
                    .or_insert_with(|| Array::zeros(param.data.raw_dim()));

                // v = momentum * v + grad
                vel.zip_mut_with(grad, |vi, &gi| {
                    *vi = self.momentum * *vi + gi;
                });

                // param -= lr * v
                let lr = self.lr;
                param.data.zip_mut_with(vel, |pi, &vi| {
                    *pi -= lr * vi;
                });
            } else {
                // Simple SGD: param -= lr * grad
                let lr = self.lr;
                param.data.zip_mut_with(grad, |pi, &gi| {
                    *pi -= lr * gi;
                });
            }
        }

        Ok(())
    }

    fn zero_state(&mut self) {
        self.velocity.clear();
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_sgd_vanilla_step() {
        let mut opt = Sgd::vanilla(0.1);
        let mut param = Parameter::new(
            "x",
            Array::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
        );
        let var_id = VarId(0);
        let grad = Array::from_shape_vec(IxDyn(&[3]), vec![0.5, 1.0, 1.5]).unwrap();
        let mut grads = HashMap::new();
        grads.insert(var_id, grad);

        opt.step(&mut [&mut param], &grads, &[var_id]).unwrap();

        // param -= 0.1 * grad
        let expected = [0.95, 1.9, 2.85];
        for (v, e) in param.data.iter().zip(expected.iter()) {
            assert!((v - e).abs() < 1e-6, "Expected {e}, got {v}");
        }
    }

    #[test]
    fn test_sgd_with_momentum() {
        let mut opt = Sgd::new(0.1, 0.9);
        let mut param = Parameter::new(
            "x",
            Array::from_elem(IxDyn(&[1]), 5.0f32),
        );
        let var_id = VarId(0);

        // Run several steps with constant gradient
        for _ in 0..50 {
            let grad = Array::from_elem(IxDyn(&[1]), 1.0f32);
            let mut grads = HashMap::new();
            grads.insert(var_id, grad);
            opt.step(&mut [&mut param], &grads, &[var_id]).unwrap();
        }

        // With momentum, we should descend faster than vanilla SGD
        // After 50 steps with momentum, param should be well below initial 5.0
        assert!(param.data[[0]] < 0.0,
            "Momentum SGD should push param below 0, got {}",
            param.data[[0]]);
    }

    #[test]
    fn test_sgd_converges_quadratic() {
        // Minimize f(x) = x^2, grad = 2x
        let mut opt = Sgd::vanilla(0.1);
        let mut param = Parameter::new(
            "x",
            Array::from_elem(IxDyn(&[1]), 3.0f32),
        );
        let var_id = VarId(0);

        for _ in 0..100 {
            let grad = param.data.mapv(|x| 2.0 * x);
            let mut grads = HashMap::new();
            grads.insert(var_id, grad);
            opt.step(&mut [&mut param], &grads, &[var_id]).unwrap();
        }

        assert!(param.data[[0]].abs() < 1e-6,
            "Should converge to 0, got {}", param.data[[0]]);
    }

    #[test]
    fn test_sgd_zero_state() {
        let mut opt = Sgd::new(0.1, 0.9);
        let mut param = Parameter::new(
            "x",
            Array::from_elem(IxDyn(&[1]), 1.0f32),
        );
        let var_id = VarId(0);
        let grad = Array::from_elem(IxDyn(&[1]), 1.0f32);
        let mut grads = HashMap::new();
        grads.insert(var_id, grad);

        opt.step(&mut [&mut param], &grads, &[var_id]).unwrap();
        assert!(!opt.velocity.is_empty());

        opt.zero_state();
        assert!(opt.velocity.is_empty());
    }

    #[test]
    fn test_sgd_lr() {
        let mut opt = Sgd::vanilla(0.01);
        assert!((opt.lr() - 0.01).abs() < 1e-10);
        opt.set_lr(0.001);
        assert!((opt.lr() - 0.001).abs() < 1e-10);
    }
}
