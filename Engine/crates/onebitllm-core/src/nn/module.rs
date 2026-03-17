use ndarray::{Array, IxDyn};

/// A named trainable parameter (full-precision latent weight).
#[derive(Debug, Clone)]
pub struct Parameter {
    /// Parameter name (for serialization, debugging).
    pub name: String,
    /// Full-precision weight data.
    pub data: Array<f32, IxDyn>,
    /// Whether this parameter requires gradient computation.
    pub requires_grad: bool,
}

impl Parameter {
    /// Create a new trainable parameter.
    pub fn new(name: impl Into<String>, data: Array<f32, IxDyn>) -> Self {
        Self {
            name: name.into(),
            data,
            requires_grad: true,
        }
    }

    /// Create a frozen (non-trainable) parameter.
    pub fn frozen(name: impl Into<String>, data: Array<f32, IxDyn>) -> Self {
        Self {
            name: name.into(),
            data,
            requires_grad: false,
        }
    }

    /// Total number of scalar elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// The core module trait. All neural network layers implement this.
///
/// Provides two forward paths:
/// - `forward_inference`: for inference with raw ndarrays (no gradient tracking).
/// - Training forward is handled via the autograd system in Phase 5.
pub trait Module: Send + Sync {
    /// Forward pass with raw ndarrays (for inference, no gradient tracking).
    fn forward_inference(&self, input: &Array<f32, IxDyn>) -> crate::Result<Array<f32, IxDyn>>;

    /// Return all trainable parameters.
    fn parameters(&self) -> Vec<&Parameter>;

    /// Return mutable references to all trainable parameters.
    fn parameters_mut(&mut self) -> Vec<&mut Parameter>;

    /// Module name (for debugging, serialization).
    fn name(&self) -> &str;

    /// Total number of scalar parameters.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}
