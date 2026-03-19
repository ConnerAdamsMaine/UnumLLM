use super::traits::Optimizer;
use crate::autograd::VarId;
use crate::nn::Parameter;
use ndarray::{Array, IxDyn};
use std::collections::HashMap;

/// AdamW optimizer with decoupled weight decay.
///
/// Implements the Adam optimizer with weight decay decoupled from the
/// gradient-based update (Loshchilov & Hutter, 2019).
pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    /// First moment estimates (per VarId).
    m: HashMap<VarId, Array<f32, IxDyn>>,
    /// Second moment estimates (per VarId).
    v: HashMap<VarId, Array<f32, IxDyn>>,
    /// Per-parameter step counter.
    t: HashMap<VarId, usize>,
}

impl AdamW {
    /// Create a new AdamW optimizer.
    ///
    /// Typical defaults: lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01.
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            m: HashMap::new(),
            v: HashMap::new(),
            t: HashMap::new(),
        }
    }

    /// Create with typical defaults.
    pub fn default_config(lr: f32) -> Self {
        Self::new(lr, 0.9, 0.999, 1e-8, 0.01)
    }
}

impl Optimizer for AdamW {
    fn step(
        &mut self,
        params: &mut [&mut Parameter],
        grads: &HashMap<VarId, Array<f32, IxDyn>>,
        param_ids: &[VarId],
    ) -> crate::Result<()> {
        if params.len() != param_ids.len() {
            return Err(crate::error::OneBitError::Training(format!(
                "params length ({}) != param_ids length ({})",
                params.len(),
                param_ids.len()
            )));
        }

        for (param, &var_id) in params.iter_mut().zip(param_ids.iter()) {
            if !param.requires_grad {
                continue;
            }

            let grad = match grads.get(&var_id) {
                Some(g) => g,
                None => continue,
            };

            // Increment step counter
            let step = self.t.entry(var_id).or_insert(0);
            *step += 1;
            let t = *step as f32;

            // Initialize moments if needed
            let m = self
                .m
                .entry(var_id)
                .or_insert_with(|| Array::zeros(param.data.raw_dim()));
            let v = self
                .v
                .entry(var_id)
                .or_insert_with(|| Array::zeros(param.data.raw_dim()));

            // Update biased first moment: m = beta1 * m + (1 - beta1) * grad
            m.zip_mut_with(grad, |mi, &gi| {
                *mi = self.beta1 * *mi + (1.0 - self.beta1) * gi;
            });

            // Update biased second moment: v = beta2 * v + (1 - beta2) * grad^2
            v.zip_mut_with(grad, |vi, &gi| {
                *vi = self.beta2 * *vi + (1.0 - self.beta2) * gi * gi;
            });

            // Bias correction
            let bc1 = 1.0 - self.beta1.powf(t);
            let bc2 = 1.0 - self.beta2.powf(t);

            // Update parameters: theta = theta - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta)
            let lr = self.lr;
            let eps = self.eps;
            let wd = self.weight_decay;

            // We need to index into m and v simultaneously with param.data
            let m_slice = m.as_slice_mut().unwrap();
            let v_slice = v.as_slice().unwrap();
            let data_slice = param.data.as_slice_mut().unwrap();

            for i in 0..data_slice.len() {
                let m_hat = m_slice[i] / bc1;
                let v_hat = v_slice[i] / bc2;
                // Decoupled weight decay + Adam update
                data_slice[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * data_slice[i]);
            }
        }

        Ok(())
    }

    fn zero_state(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t.clear();
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
    fn test_adamw_converges_quadratic() {
        // Minimize f(x) = sum(x^2), optimal at x = 0
        let mut opt = AdamW::default_config(0.1);

        let mut param = Parameter::new(
            "x",
            Array::from_shape_vec(IxDyn(&[4]), vec![5.0, -3.0, 2.0, -1.0]).unwrap(),
        );
        let var_id = VarId(0);

        for _ in 0..200 {
            // grad of sum(x^2) = 2*x
            let grad = param.data.mapv(|x| 2.0 * x);
            let mut grads = HashMap::new();
            grads.insert(var_id, grad);

            opt.step(&mut [&mut param], &grads, &[var_id]).unwrap();
        }

        // All values should be near zero
        for &v in param.data.iter() {
            assert!(v.abs() < 0.1, "Expected near-zero, got {v}");
        }
    }

    #[test]
    fn test_adamw_zero_state() {
        let mut opt = AdamW::default_config(0.01);
        let var_id = VarId(0);
        let mut param = Parameter::new("x", Array::from_elem(IxDyn(&[2]), 1.0f32));
        let grad = Array::from_elem(IxDyn(&[2]), 0.5f32);
        let mut grads = HashMap::new();
        grads.insert(var_id, grad);

        opt.step(&mut [&mut param], &grads, &[var_id]).unwrap();
        assert!(!opt.m.is_empty());

        opt.zero_state();
        assert!(opt.m.is_empty());
        assert!(opt.v.is_empty());
        assert!(opt.t.is_empty());
    }

    #[test]
    fn test_adamw_lr() {
        let mut opt = AdamW::default_config(0.01);
        assert!((opt.lr() - 0.01).abs() < 1e-10);
        opt.set_lr(0.001);
        assert!((opt.lr() - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_adamw_skips_frozen_params() {
        let mut opt = AdamW::default_config(0.1);
        let mut param = Parameter::frozen("frozen", Array::from_elem(IxDyn(&[2]), 1.0f32));
        let original = param.data.clone();
        let var_id = VarId(0);

        let grad = Array::from_elem(IxDyn(&[2]), 1.0f32);
        let mut grads = HashMap::new();
        grads.insert(var_id, grad);

        opt.step(&mut [&mut param], &grads, &[var_id]).unwrap();

        // Frozen param should not change
        assert_eq!(param.data, original);
    }

    #[test]
    fn test_adamw_weight_decay_effect() {
        // Compare optimizer with and without weight decay
        let mut opt_wd = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.1);
        let mut opt_no_wd = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.0);

        let mut param_wd = Parameter::new("wd", Array::from_elem(IxDyn(&[1]), 5.0f32));
        let mut param_no_wd = Parameter::new("no_wd", Array::from_elem(IxDyn(&[1]), 5.0f32));
        let var_id = VarId(0);

        for _ in 0..50 {
            let grad = Array::from_elem(IxDyn(&[1]), 0.1f32);
            let mut grads = HashMap::new();
            grads.insert(var_id, grad);

            opt_wd
                .step(&mut [&mut param_wd], &grads, &[var_id])
                .unwrap();
            opt_no_wd
                .step(&mut [&mut param_no_wd], &grads, &[var_id])
                .unwrap();
        }

        // With weight decay, the param should be smaller (more regularized)
        assert!(
            param_wd.data[[0]] < param_no_wd.data[[0]],
            "Weight decay should push params toward zero: wd={} no_wd={}",
            param_wd.data[[0]],
            param_no_wd.data[[0]],
        );
    }
}
