use std::collections::HashMap;
use ndarray::{Array, IxDyn};
use crate::nn::Parameter;
use crate::autograd::VarId;

/// Trait for parameter optimizers (AdamW, SGD, etc.).
///
/// The optimizer receives the computed gradients as a map from `VarId` to
/// gradient arrays, and updates the parameter data in-place.
pub trait Optimizer {
    /// Perform one optimization step: update `params` using the given `grads`.
    ///
    /// - `params`: mutable slice of model parameters to update.
    /// - `grads`: map from variable ID to gradient array.
    /// - `param_ids`: parallel slice mapping each parameter to its VarId on
    ///   the tape (same order as `params`).
    fn step(
        &mut self,
        params: &mut [&mut Parameter],
        grads: &HashMap<VarId, Array<f32, IxDyn>>,
        param_ids: &[VarId],
    ) -> crate::Result<()>;

    /// Reset all internal optimizer state (moments, step counters, etc.).
    fn zero_state(&mut self);

    /// Get the current learning rate.
    fn lr(&self) -> f32;

    /// Set the learning rate.
    fn set_lr(&mut self, lr: f32);
}
