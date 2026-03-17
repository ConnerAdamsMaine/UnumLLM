//! Training loop abstractions.
//!
//! Provides a `Trainer` struct that orchestrates the forward pass, backward
//! pass, and optimizer step for each training iteration.

use std::collections::HashMap;
use ndarray::{Array, IxDyn};
use crate::autograd::{Tape, Variable, VarId};
use crate::nn::Parameter;
use crate::optim::traits::Optimizer;

/// Configuration for a training run.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Maximum number of training steps.
    pub max_steps: usize,
    /// Logging interval (every N steps).
    pub log_interval: usize,
    /// Gradient clipping max norm (0.0 = disabled).
    pub grad_clip_norm: f32,
    /// Whether to use QAT (quantization-aware training).
    pub use_qat: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            max_steps: 1000,
            log_interval: 100,
            grad_clip_norm: 1.0,
            use_qat: true,
        }
    }
}

/// Result of a single training step.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// The loss value for this step.
    pub loss: f32,
    /// The gradient norm before clipping (if clipping is enabled).
    pub grad_norm: f32,
}

/// A trainer that manages the training loop.
///
/// The trainer creates a fresh tape for each forward pass, runs backward,
/// optionally clips gradients, and calls the optimizer.
pub struct Trainer {
    config: TrainConfig,
    step_count: usize,
}

impl Trainer {
    /// Create a new trainer with the given configuration.
    pub fn new(config: TrainConfig) -> Self {
        Self {
            config,
            step_count: 0,
        }
    }

    /// Get the current step count.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get the training configuration.
    pub fn config(&self) -> &TrainConfig {
        &self.config
    }

    /// Perform a single training step.
    ///
    /// This is a lower-level method that takes a closure for the forward pass.
    /// The closure receives:
    /// - A reference to the tape
    /// - The parameter variables (one per parameter, in order)
    ///
    /// It must return the scalar loss `Variable`.
    ///
    /// Returns the step result with loss and gradient norm.
    pub fn train_step<F>(
        &mut self,
        params: &mut [&mut Parameter],
        optimizer: &mut dyn Optimizer,
        forward_fn: F,
    ) -> crate::Result<StepResult>
    where
        F: FnOnce(&std::sync::Arc<std::sync::Mutex<crate::autograd::tape::Tape>>, &[Variable]) -> crate::Result<Variable>,
    {
        // 1. Create a fresh tape.
        let tape = Tape::new();

        // 2. Create variables for each parameter.
        let mut param_vars = Vec::with_capacity(params.len());
        let mut param_ids = Vec::with_capacity(params.len());
        for param in params.iter() {
            let var = Variable::new(param.data.clone(), param.requires_grad, &tape);
            param_ids.push(var.id);
            param_vars.push(var);
        }

        // 3. Run the forward pass.
        let loss_var = forward_fn(&tape, &param_vars)?;
        let loss_val = loss_var.data.iter().copied().sum::<f32>();

        // 4. Run backward.
        let grads = loss_var.backward()?;

        // 5. Compute gradient norm and optionally clip.
        let grad_norm = compute_grad_norm(&grads, &param_ids);
        let grads = if self.config.grad_clip_norm > 0.0 && grad_norm > self.config.grad_clip_norm {
            clip_gradients(grads, &param_ids, self.config.grad_clip_norm, grad_norm)
        } else {
            grads
        };

        // 6. Optimizer step.
        optimizer.step(params, &grads, &param_ids)?;

        self.step_count += 1;

        Ok(StepResult {
            loss: loss_val,
            grad_norm,
        })
    }
}

/// Compute the L2 norm of all gradients for the given parameter IDs.
fn compute_grad_norm(
    grads: &HashMap<VarId, Array<f32, IxDyn>>,
    param_ids: &[VarId],
) -> f32 {
    let mut norm_sq = 0.0f32;
    for id in param_ids {
        if let Some(g) = grads.get(id) {
            norm_sq += g.iter().map(|v| v * v).sum::<f32>();
        }
    }
    norm_sq.sqrt()
}

/// Clip gradients by global norm.
fn clip_gradients(
    mut grads: HashMap<VarId, Array<f32, IxDyn>>,
    param_ids: &[VarId],
    max_norm: f32,
    current_norm: f32,
) -> HashMap<VarId, Array<f32, IxDyn>> {
    let scale = max_norm / (current_norm + 1e-6);
    for id in param_ids {
        if let Some(g) = grads.get_mut(id) {
            g.mapv_inplace(|v| v * scale);
        }
    }
    grads
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::ops;
    use crate::optim::sgd::Sgd;
    use ndarray::Array;

    #[test]
    fn test_train_config_default() {
        let cfg = TrainConfig::default();
        assert_eq!(cfg.max_steps, 1000);
        assert_eq!(cfg.log_interval, 100);
        assert!(cfg.use_qat);
    }

    #[test]
    fn test_trainer_step_count() {
        let mut trainer = Trainer::new(TrainConfig::default());
        assert_eq!(trainer.step_count(), 0);

        let mut param = Parameter::new(
            "x",
            Array::from_elem(IxDyn(&[2]), 1.0f32),
        );
        let mut opt = Sgd::vanilla(0.1);

        let result = trainer.train_step(
            &mut [&mut param],
            &mut opt,
            |_tape, vars| {
                Ok(ops::sum(&vars[0]))
            },
        ).unwrap();

        assert_eq!(trainer.step_count(), 1);
        assert!(result.loss > 0.0);
    }

    #[test]
    fn test_trainer_minimizes_quadratic() {
        // f(x) = sum(x * x), minimum at x = 0
        let mut trainer = Trainer::new(TrainConfig {
            grad_clip_norm: 0.0, // disable clipping for this test
            ..TrainConfig::default()
        });
        let mut param = Parameter::new(
            "x",
            Array::from_shape_vec(IxDyn(&[3]), vec![3.0, -2.0, 1.0]).unwrap(),
        );
        let mut opt = Sgd::vanilla(0.1);

        for _ in 0..100 {
            let _result = trainer.train_step(
                &mut [&mut param],
                &mut opt,
                |_tape, vars| {
                    let x = &vars[0];
                    let x2 = ops::mul(x, x);
                    Ok(ops::sum(&x2))
                },
            ).unwrap();
        }

        // Should converge near zero
        for v in param.data.iter() {
            assert!(v.abs() < 0.1, "Expected near-zero, got {v}");
        }
    }

    #[test]
    fn test_gradient_clipping() {
        let grads = {
            let mut m = HashMap::new();
            // Gradient with norm = 5.0 (3^2 + 4^2 = 25, sqrt = 5)
            m.insert(VarId(0), Array::from_shape_vec(IxDyn(&[2]), vec![3.0, 4.0]).unwrap());
            m
        };
        let ids = [VarId(0)];
        let norm = compute_grad_norm(&grads, &ids);
        assert!((norm - 5.0).abs() < 1e-5);

        let clipped = clip_gradients(grads, &ids, 1.0, norm);
        let clipped_norm = compute_grad_norm(&clipped, &ids);
        assert!((clipped_norm - 1.0).abs() < 1e-4,
            "Clipped norm should be ~1.0, got {clipped_norm}");
    }
}
